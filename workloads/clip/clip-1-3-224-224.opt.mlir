module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot.2_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.100_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.101_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.102_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.103_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.104_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.4_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.108_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.109_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.110_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.111_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.112_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.6_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.116_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.117_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.118_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.119_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.120_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.8_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.124_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.125_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.126_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.127_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.128_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.10_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.132_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.133_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.134_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.135_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.136_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.12_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.140_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.141_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.142_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.143_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.144_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.14_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.148_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.149_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.150_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.151_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.152_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.16_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.156_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.157_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.158_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.159_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.160_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.18_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.164_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.165_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.166_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.167_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.168_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.20_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.172_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.173_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.174_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.175_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.176_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.22_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.180_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.181_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.182_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.183_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.184_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot.24_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768x768xf32>) -> tensor<2304x50xf32> {
    %0 = stablehlo.concatenate %arg3, %arg4, %arg5, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.multiply %arg0, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %5 = stablehlo.add %3, %4 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %6 = stablehlo.dot_general %1, %5, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
    return %6 : tensor<2304x50xf32>
  }
  func.func private @gemm_fusion_dot_general.188_computation(%arg0: tensor<12x64xf32>, %arg1: tensor<12x64x50xf32>, %arg2: tensor<12x64xf32>, %arg3: tensor<12x64x50xf32>) -> tensor<12x50x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<12x64x50xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %6 = stablehlo.add %5, %arg3 : tensor<12x64x50xf32>
    %7 = stablehlo.multiply %6, %2 : tensor<12x64x50xf32>
    %8 = stablehlo.dot_general %4, %7, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    return %8 : tensor<12x50x50xf32>
  }
  func.func private @gemm_fusion_dot_general.189_computation(%arg0: tensor<12x50x50xf32>, %arg1: tensor<12x64xf32>, %arg2: tensor<12x64x50xf32>) -> tensor<50x64x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<12x64x50xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
    %3 = stablehlo.dot_general %2, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    return %4 : tensor<50x64x12xf32>
  }
  func.func private @gemm_fusion_dot_general.190_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768x768xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @gemm_fusion_dot_general.191_computation(%arg0: tensor<50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
    return %4 : tensor<3072x50xf32>
  }
  func.func private @gemm_fusion_dot_general.192_computation(%arg0: tensor<50x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @region_1.4.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    %4 = stablehlo.slice %3 [0:1, 0:768, 1:50] : (tensor<1x768x50xf32>) -> tensor<1x768x49xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x49xf32>) -> tensor<768x49xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<768x49xf32>, tensor<f32>) -> tensor<768xf32>
    return %6 : tensor<768xf32>
  }
  func.func private @region_1.4.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @region_1.4.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x50xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %2 : tensor<50xf32>
  }
  func.func private @region_1.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<1x768x50xf32>) -> tensor<50xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
    return %1 : tensor<50xf32>
  }
  func.func private @fused_add(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768x50xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x50xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x50xf32>
    return %3 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<3072x50xf32>, %arg1: tensor<3072xf32>) -> tensor<1x50x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
    %2 = stablehlo.add %0, %1 : tensor<50x3072xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
    %6 = stablehlo.multiply %3, %5 : tensor<1x50x3072xf32>
    %7 = stablehlo.negate %6 : tensor<1x50x3072xf32>
    %8 = stablehlo.exponential %7 : tensor<1x50x3072xf32>
    %9 = stablehlo.add %4, %8 : tensor<1x50x3072xf32>
    %10 = stablehlo.divide %4, %9 : tensor<1x50x3072xf32>
    %11 = stablehlo.multiply %3, %10 : tensor<1x50x3072xf32>
    return %11 : tensor<1x50x3072xf32>
  }
  func.func private @region_1.4.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.241(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.242(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.243(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.244(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.245(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.246(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.247(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.248(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.249(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.250(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.251(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @region_1.4.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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
  func.func private @fused_computation.252(%arg0: tensor<12x50x50xf32>) -> tensor<12x50x50xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %1 = stablehlo.compare EQ, %arg0, %0 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
    %2 = stablehlo.not %1 : tensor<12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %3 = stablehlo.reduce(%2 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
    %4 = stablehlo.not %3 : tensor<12x50xi1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
    %7 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %9 = stablehlo.subtract %arg0, %8 : tensor<12x50x50xf32>
    %10 = stablehlo.exponential %9 : tensor<12x50x50xf32>
    %11 = stablehlo.reduce(%10 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
    %13 = stablehlo.divide %10, %12 : tensor<12x50x50xf32>
    %14 = stablehlo.select %5, %6, %13 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
    return %14 : tensor<12x50x50xf32>
  }
  func.func private @fused_concatenate(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<1x50xi32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768x1xf32>
    %1 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<1x768x49xf32>
    %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<1x768x1xf32>, tensor<1x768x49xf32>) -> tensor<1x768x50xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %4 = stablehlo.compare LT, %arg3, %3 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %6 = stablehlo.add %arg3, %5 : tensor<1x50xi32>
    %7 = stablehlo.select %4, %6, %arg3 : tensor<1x50xi1>, tensor<1x50xi32>
    %8 = stablehlo.compare GE, %7, %3 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %10 = stablehlo.compare LE, %7, %9 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %11 = stablehlo.and %8, %10 : tensor<1x50xi1>
    %12 = mhlo.bitcast %11 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %13 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<50xi1>) -> tensor<1x768x50xi1>
    %14 = mhlo.bitcast %7 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
    %15 = "stablehlo.gather"(%arg2, %14) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[50,1,768]{0,2,1}"} : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 2, 1]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x768x50xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768x50xf32>
    %18 = stablehlo.select %13, %16, %17 : tensor<1x768x50xi1>, tensor<1x768x50xf32>
    %19 = stablehlo.add %2, %18 : tensor<1x768x50xf32>
    return %19 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<1x768x50xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<1x768x50xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x768x50xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1x768x50xf32>
    %12 = stablehlo.add %10, %11 : tensor<1x768x50xf32>
    return %12 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.1(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.2(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.3(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.4(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.5(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.6(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.7(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.8(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.9(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.10(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.11(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.12(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.13(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.14(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.15(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.16(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.17(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.18(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.19(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.20(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.21(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.22(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.23(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @fused_subtract.24(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x50xf32>
    return %5 : tensor<1x768x50xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<1x768x50xf32>, %arg1: tensor<50xf32>) -> tensor<1x768x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x50xf32>
    return %8 : tensor<1x768x50xf32>
  }
  func.func private @region_1.4.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<2x768xf32>, %arg1: tensor<768xf32>) -> tensor<2xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<768xf32>) -> tensor<2x768xf32>
    %5 = stablehlo.multiply %4, %arg0 : tensor<2x768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<2x768xf32>, tensor<f32>) -> tensor<2xf32>
    return %6 : tensor<2xf32>
  }
  func.func private @wrapped_slice_computation(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.1(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.2(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.3(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.4(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.5(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.6(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.7(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.8(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.9(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.10(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.11(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.12(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.13(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.14(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.15(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.16(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.17(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.18(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.19(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.20(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.21(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.22(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.23(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.24(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.25(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.26(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.27(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.28(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.29(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.30(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.31(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.32(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.33(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.34(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_slice_computation.35(%arg0: tensor<2304x50xf32>) -> tensor<768x50xf32> {
    %0 = stablehlo.slice %arg0 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    return %0 : tensor<768x50xf32>
  }
  func.func private @wrapped_add_computation(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
  func.func @main(%arg0: tensor<768xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<3072xf32>, %arg33: tensor<768x3072xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<3072x768xf32>, %arg48: tensor<3072xf32>, %arg49: tensor<768x3072xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<3072x768xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<768x3072xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<3072x768xf32>, %arg80: tensor<3072xf32>, %arg81: tensor<768x3072xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<3072x768xf32>, %arg96: tensor<3072xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<3072xf32>, %arg113: tensor<768x3072xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<3072x768xf32>, %arg128: tensor<3072xf32>, %arg129: tensor<768x3072xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768x768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<3072x768xf32>, %arg144: tensor<3072xf32>, %arg145: tensor<768x3072xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768x768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<3072x768xf32>, %arg160: tensor<3072xf32>, %arg161: tensor<768x3072xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<3072x768xf32>, %arg176: tensor<3072xf32>, %arg177: tensor<768x3072xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<768x3072xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<2x768xf32>, %arg198: tensor<2xf32>, %arg199: tensor<1x50xi32>, %arg200: tensor<1x3x224x224xf32>) -> tensor<1x2xf32> {
    %0 = stablehlo.custom_call @__cudnn$convForward(%arg200, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x224x224xf32>, tensor<768x3x32x32xf32>) -> tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>) -> tensor<1x768x7x7xf32>
    %2 = "mhlo.fusion"(%1, %arg0, %arg2, %arg199) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768xf32>, %arg203: tensor<50x768xf32>, %arg204: tensor<1x50xi32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768x1xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<1x768x49xf32>
      %384 = stablehlo.concatenate %382, %383, dim = 2 : (tensor<1x768x1xf32>, tensor<1x768x49xf32>) -> tensor<1x768x50xf32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %385 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %386 = stablehlo.compare LT, %arg204, %385 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_0 = stablehlo.constant dense<50> : tensor<i32>
      %387 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %388 = stablehlo.add %arg204, %387 : tensor<1x50xi32>
      %389 = stablehlo.select %386, %388, %arg204 : tensor<1x50xi1>, tensor<1x50xi32>
      %390 = stablehlo.compare GE, %389, %385 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_1 = stablehlo.constant dense<49> : tensor<i32>
      %391 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %392 = stablehlo.compare LE, %389, %391 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %393 = stablehlo.and %390, %392 : tensor<1x50xi1>
      %394 = mhlo.bitcast %393 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %395 = stablehlo.broadcast_in_dim %394, dims = [2] : (tensor<50xi1>) -> tensor<1x768x50xi1>
      %396 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %397 = "stablehlo.gather"(%arg203, %396) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> {result_layout = dense<[0, 2, 1]> : tensor<3xindex>, xla_shape = "f32[50,1,768]{0,2,1}"} : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      %398 = mhlo.bitcast %397 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 2, 1]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x768x50xf32>
      %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768x50xf32>
      %400 = stablehlo.select %395, %398, %399 : tensor<1x768x50xi1>, tensor<1x768x50xf32>
      %401 = stablehlo.add %384, %400 : tensor<1x768x50xf32>
      mhlo.return %401 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768xf32>, tensor<50x768xf32>, tensor<1x50xi32>) -> tensor<1x768x50xf32>
    %3 = "mhlo.fusion"(%2) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %4 = "mhlo.fusion"(%2, %3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %5 = "mhlo.fusion"(%4) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %6 = "mhlo.fusion"(%4, %arg4, %arg3, %5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<50xf32>):
      %382 = mhlo.bitcast %arg204 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      %391 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<1x768x50xf32>
      %392 = stablehlo.multiply %390, %391 : tensor<1x768x50xf32>
      %393 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1x768x50xf32>
      %394 = stablehlo.add %392, %393 : tensor<1x768x50xf32>
      mhlo.return %394 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %7 = mhlo.bitcast %arg10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %8 = "mhlo.fusion"(%6) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %9 = "mhlo.fusion"(%6, %8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %10 = "mhlo.fusion"(%9) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %11 = "mhlo.fusion"(%9, %10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %13 = "mhlo.fusion"(%12, %arg13, %arg14, %arg9, %arg5, %arg7) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %14 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %15 = mhlo.bitcast %14 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %16 = mhlo.bitcast %arg6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %17 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %19 = "mhlo.fusion"(%7, %15, %16, %18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %20 = "mhlo.fusion"(%19) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %21 = mhlo.bitcast %arg8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %22 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %24 = "mhlo.fusion"(%20, %21, %23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %25 = mhlo.bitcast %24 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %26 = "mhlo.fusion"(%25, %arg11) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %27 = "mhlo.fusion"(%6, %26, %arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %28 = "mhlo.fusion"(%27) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %29 = "mhlo.fusion"(%27, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %30 = "mhlo.fusion"(%29) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %31 = "mhlo.fusion"(%29, %30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %32 = mhlo.bitcast %31 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %33 = "mhlo.fusion"(%32, %arg19, %arg20, %arg15) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %34 = "mhlo.fusion"(%33, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %35 = mhlo.bitcast %34 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %36 = "mhlo.fusion"(%35, %arg17) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %37 = "mhlo.fusion"(%27, %36, %arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %38 = mhlo.bitcast %arg26 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %39 = "mhlo.fusion"(%37) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %40 = "mhlo.fusion"(%37, %39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %42 = "mhlo.fusion"(%40, %41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %43 = mhlo.bitcast %42 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %44 = "mhlo.fusion"(%43, %arg29, %arg30, %arg25, %arg21, %arg23) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %45 = "mhlo.fusion"(%44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %46 = mhlo.bitcast %45 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %47 = mhlo.bitcast %arg22 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %48 = "mhlo.fusion"(%44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %49 = mhlo.bitcast %48 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %50 = "mhlo.fusion"(%38, %46, %47, %49) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %51 = "mhlo.fusion"(%50) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %52 = mhlo.bitcast %arg24 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %53 = "mhlo.fusion"(%44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %54 = mhlo.bitcast %53 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %55 = "mhlo.fusion"(%51, %52, %54) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %57 = "mhlo.fusion"(%56, %arg27) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %58 = "mhlo.fusion"(%37, %57, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %59 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %60 = "mhlo.fusion"(%58, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %62 = "mhlo.fusion"(%60, %61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %64 = "mhlo.fusion"(%63, %arg35, %arg36, %arg31) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %65 = "mhlo.fusion"(%64, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %66 = mhlo.bitcast %65 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %67 = "mhlo.fusion"(%66, %arg33) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %68 = "mhlo.fusion"(%58, %67, %arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %69 = mhlo.bitcast %arg42 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %70 = "mhlo.fusion"(%68) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %71 = "mhlo.fusion"(%68, %70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %72 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %75 = "mhlo.fusion"(%74, %arg45, %arg46, %arg41, %arg37, %arg39) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %76 = "mhlo.fusion"(%75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %77 = mhlo.bitcast %76 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %78 = mhlo.bitcast %arg38 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %79 = "mhlo.fusion"(%75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %81 = "mhlo.fusion"(%69, %77, %78, %80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %82 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %83 = mhlo.bitcast %arg40 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %84 = "mhlo.fusion"(%75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %85 = mhlo.bitcast %84 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %86 = "mhlo.fusion"(%82, %83, %85) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %87 = mhlo.bitcast %86 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %88 = "mhlo.fusion"(%87, %arg43) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %89 = "mhlo.fusion"(%68, %88, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %91 = "mhlo.fusion"(%89, %90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %92 = "mhlo.fusion"(%91) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %93 = "mhlo.fusion"(%91, %92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %94 = mhlo.bitcast %93 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %95 = "mhlo.fusion"(%94, %arg51, %arg52, %arg47) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %96 = "mhlo.fusion"(%95, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %98 = "mhlo.fusion"(%97, %arg49) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %99 = "mhlo.fusion"(%89, %98, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %100 = mhlo.bitcast %arg58 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %101 = "mhlo.fusion"(%99) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %102 = "mhlo.fusion"(%99, %101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %103 = "mhlo.fusion"(%102) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %104 = "mhlo.fusion"(%102, %103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %105 = mhlo.bitcast %104 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %106 = "mhlo.fusion"(%105, %arg61, %arg62, %arg57, %arg53, %arg55) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %107 = "mhlo.fusion"(%106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %108 = mhlo.bitcast %107 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %109 = mhlo.bitcast %arg54 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %110 = "mhlo.fusion"(%106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %111 = mhlo.bitcast %110 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %112 = "mhlo.fusion"(%100, %108, %109, %111) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %113 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %114 = mhlo.bitcast %arg56 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %115 = "mhlo.fusion"(%106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %117 = "mhlo.fusion"(%113, %114, %116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %118 = mhlo.bitcast %117 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %119 = "mhlo.fusion"(%118, %arg59) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %120 = "mhlo.fusion"(%99, %119, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %121 = "mhlo.fusion"(%120) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %122 = "mhlo.fusion"(%120, %121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %123 = "mhlo.fusion"(%122) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %124 = "mhlo.fusion"(%122, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %125 = mhlo.bitcast %124 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %126 = "mhlo.fusion"(%125, %arg67, %arg68, %arg63) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %127 = "mhlo.fusion"(%126, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %128 = mhlo.bitcast %127 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %129 = "mhlo.fusion"(%128, %arg65) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %130 = "mhlo.fusion"(%120, %129, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %131 = mhlo.bitcast %arg74 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %132 = "mhlo.fusion"(%130) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %133 = "mhlo.fusion"(%130, %132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %135 = "mhlo.fusion"(%133, %134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %137 = "mhlo.fusion"(%136, %arg77, %arg78, %arg73, %arg69, %arg71) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %138 = "mhlo.fusion"(%137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %139 = mhlo.bitcast %138 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %140 = mhlo.bitcast %arg70 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %141 = "mhlo.fusion"(%137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %142 = mhlo.bitcast %141 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %143 = "mhlo.fusion"(%131, %139, %140, %142) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %144 = "mhlo.fusion"(%143) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %145 = mhlo.bitcast %arg72 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %146 = "mhlo.fusion"(%137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %147 = mhlo.bitcast %146 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %148 = "mhlo.fusion"(%144, %145, %147) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %149 = mhlo.bitcast %148 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %150 = "mhlo.fusion"(%149, %arg75) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %151 = "mhlo.fusion"(%130, %150, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %152 = "mhlo.fusion"(%151) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %153 = "mhlo.fusion"(%151, %152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %155 = "mhlo.fusion"(%153, %154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %156 = mhlo.bitcast %155 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %157 = "mhlo.fusion"(%156, %arg83, %arg84, %arg79) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %158 = "mhlo.fusion"(%157, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %159 = mhlo.bitcast %158 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %160 = "mhlo.fusion"(%159, %arg81) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %161 = "mhlo.fusion"(%151, %160, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %162 = mhlo.bitcast %arg90 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %163 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %164 = "mhlo.fusion"(%161, %163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %166 = "mhlo.fusion"(%164, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %167 = mhlo.bitcast %166 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %168 = "mhlo.fusion"(%167, %arg93, %arg94, %arg89, %arg85, %arg87) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %170 = mhlo.bitcast %169 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %171 = mhlo.bitcast %arg86 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %172 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %173 = mhlo.bitcast %172 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %174 = "mhlo.fusion"(%162, %170, %171, %173) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %176 = mhlo.bitcast %arg88 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %177 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %178 = mhlo.bitcast %177 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %179 = "mhlo.fusion"(%175, %176, %178) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %180 = mhlo.bitcast %179 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %181 = "mhlo.fusion"(%180, %arg91) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %182 = "mhlo.fusion"(%161, %181, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %184 = "mhlo.fusion"(%182, %183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %185 = "mhlo.fusion"(%184) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %186 = "mhlo.fusion"(%184, %185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %187 = mhlo.bitcast %186 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %188 = "mhlo.fusion"(%187, %arg99, %arg100, %arg95) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %189 = "mhlo.fusion"(%188, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %190 = mhlo.bitcast %189 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %191 = "mhlo.fusion"(%190, %arg97) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %192 = "mhlo.fusion"(%182, %191, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %193 = mhlo.bitcast %arg106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %194 = "mhlo.fusion"(%192) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %195 = "mhlo.fusion"(%192, %194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %197 = "mhlo.fusion"(%195, %196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %198 = mhlo.bitcast %197 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %199 = "mhlo.fusion"(%198, %arg109, %arg110, %arg105, %arg101, %arg103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %200 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %202 = mhlo.bitcast %arg102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %203 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %204 = mhlo.bitcast %203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %205 = "mhlo.fusion"(%193, %201, %202, %204) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %206 = "mhlo.fusion"(%205) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %207 = mhlo.bitcast %arg104 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %208 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %209 = mhlo.bitcast %208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %210 = "mhlo.fusion"(%206, %207, %209) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %211 = mhlo.bitcast %210 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %212 = "mhlo.fusion"(%211, %arg107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %213 = "mhlo.fusion"(%192, %212, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %214 = "mhlo.fusion"(%213) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %215 = "mhlo.fusion"(%213, %214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %216 = "mhlo.fusion"(%215) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %217 = "mhlo.fusion"(%215, %216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %219 = "mhlo.fusion"(%218, %arg115, %arg116, %arg111) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %220 = "mhlo.fusion"(%219, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %221 = mhlo.bitcast %220 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %222 = "mhlo.fusion"(%221, %arg113) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %223 = "mhlo.fusion"(%213, %222, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %224 = mhlo.bitcast %arg122 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %225 = "mhlo.fusion"(%223) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %226 = "mhlo.fusion"(%223, %225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %227 = "mhlo.fusion"(%226) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %228 = "mhlo.fusion"(%226, %227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %230 = "mhlo.fusion"(%229, %arg125, %arg126, %arg121, %arg117, %arg119) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %231 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %232 = mhlo.bitcast %231 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %233 = mhlo.bitcast %arg118 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %234 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %235 = mhlo.bitcast %234 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %236 = "mhlo.fusion"(%224, %232, %233, %235) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %237 = "mhlo.fusion"(%236) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %238 = mhlo.bitcast %arg120 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %239 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %240 = mhlo.bitcast %239 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %241 = "mhlo.fusion"(%237, %238, %240) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %242 = mhlo.bitcast %241 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %243 = "mhlo.fusion"(%242, %arg123) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %244 = "mhlo.fusion"(%223, %243, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %245 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %246 = "mhlo.fusion"(%244, %245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %248 = "mhlo.fusion"(%246, %247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %249 = mhlo.bitcast %248 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %250 = "mhlo.fusion"(%249, %arg131, %arg132, %arg127) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %251 = "mhlo.fusion"(%250, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %253 = "mhlo.fusion"(%252, %arg129) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %254 = "mhlo.fusion"(%244, %253, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %255 = mhlo.bitcast %arg138 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %256 = "mhlo.fusion"(%254) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %257 = "mhlo.fusion"(%254, %256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %259 = "mhlo.fusion"(%257, %258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %260 = mhlo.bitcast %259 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %261 = "mhlo.fusion"(%260, %arg141, %arg142, %arg137, %arg133, %arg135) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %262 = "mhlo.fusion"(%261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %263 = mhlo.bitcast %262 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %264 = mhlo.bitcast %arg134 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %265 = "mhlo.fusion"(%261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %266 = mhlo.bitcast %265 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %267 = "mhlo.fusion"(%255, %263, %264, %266) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %268 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %269 = mhlo.bitcast %arg136 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %270 = "mhlo.fusion"(%261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %271 = mhlo.bitcast %270 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %272 = "mhlo.fusion"(%268, %269, %271) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %274 = "mhlo.fusion"(%273, %arg139) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %275 = "mhlo.fusion"(%254, %274, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %276 = "mhlo.fusion"(%275) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %277 = "mhlo.fusion"(%275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %278 = "mhlo.fusion"(%277) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %279 = "mhlo.fusion"(%277, %278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %280 = mhlo.bitcast %279 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %281 = "mhlo.fusion"(%280, %arg147, %arg148, %arg143) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %282 = "mhlo.fusion"(%281, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %283 = mhlo.bitcast %282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %284 = "mhlo.fusion"(%283, %arg145) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %285 = "mhlo.fusion"(%275, %284, %arg146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %286 = mhlo.bitcast %arg154 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %287 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %288 = "mhlo.fusion"(%285, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %289 = "mhlo.fusion"(%288) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %290 = "mhlo.fusion"(%288, %289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %291 = mhlo.bitcast %290 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %292 = "mhlo.fusion"(%291, %arg157, %arg158, %arg153, %arg149, %arg151) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %293 = "mhlo.fusion"(%292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %295 = mhlo.bitcast %arg150 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %296 = "mhlo.fusion"(%292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %298 = "mhlo.fusion"(%286, %294, %295, %297) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %299 = "mhlo.fusion"(%298) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %300 = mhlo.bitcast %arg152 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %301 = "mhlo.fusion"(%292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %302 = mhlo.bitcast %301 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %303 = "mhlo.fusion"(%299, %300, %302) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %305 = "mhlo.fusion"(%304, %arg155) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %306 = "mhlo.fusion"(%285, %305, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %307 = "mhlo.fusion"(%306) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %308 = "mhlo.fusion"(%306, %307) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %309 = "mhlo.fusion"(%308) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %310 = "mhlo.fusion"(%308, %309) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %311 = mhlo.bitcast %310 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %312 = "mhlo.fusion"(%311, %arg163, %arg164, %arg159) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %313 = "mhlo.fusion"(%312, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %314 = mhlo.bitcast %313 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %315 = "mhlo.fusion"(%314, %arg161) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %316 = "mhlo.fusion"(%306, %315, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %317 = mhlo.bitcast %arg170 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %318 = "mhlo.fusion"(%316) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %319 = "mhlo.fusion"(%316, %318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %320 = "mhlo.fusion"(%319) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %321 = "mhlo.fusion"(%319, %320) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %322 = mhlo.bitcast %321 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %323 = "mhlo.fusion"(%322, %arg173, %arg174, %arg169, %arg165, %arg167) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %324 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %325 = mhlo.bitcast %324 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %326 = mhlo.bitcast %arg166 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %327 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %328 = mhlo.bitcast %327 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %329 = "mhlo.fusion"(%317, %325, %326, %328) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %330 = "mhlo.fusion"(%329) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %331 = mhlo.bitcast %arg168 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %332 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %333 = mhlo.bitcast %332 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %334 = "mhlo.fusion"(%330, %331, %333) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %335 = mhlo.bitcast %334 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %336 = "mhlo.fusion"(%335, %arg171) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %337 = "mhlo.fusion"(%316, %336, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %338 = "mhlo.fusion"(%337) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %339 = "mhlo.fusion"(%337, %338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %340 = "mhlo.fusion"(%339) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %341 = "mhlo.fusion"(%339, %340) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %342 = mhlo.bitcast %341 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %343 = "mhlo.fusion"(%342, %arg179, %arg180, %arg175) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %344 = "mhlo.fusion"(%343, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %345 = mhlo.bitcast %344 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %346 = "mhlo.fusion"(%345, %arg177) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %347 = "mhlo.fusion"(%337, %346, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %348 = mhlo.bitcast %arg186 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %349 = "mhlo.fusion"(%347) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %350 = "mhlo.fusion"(%347, %349) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %351 = "mhlo.fusion"(%350) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %352 = "mhlo.fusion"(%350, %351) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %353 = mhlo.bitcast %352 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %354 = "mhlo.fusion"(%353, %arg189, %arg190, %arg185, %arg181, %arg183) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768x768xf32>, %arg205: tensor<768x768xf32>, %arg206: tensor<768x768xf32>):
      %382 = stablehlo.concatenate %arg204, %arg205, %arg206, dim = 0 : (tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x768xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[768,2304]{0,1}"} : (tensor<2304x768xf32>) -> tensor<768x2304xf32>
      %384 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.multiply %arg201, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %387 = stablehlo.add %385, %386 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %388 = stablehlo.dot_general %383, %387, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x2304xf32>, tensor<50x768xf32>) -> tensor<2304x50xf32>
      mhlo.return %388 : tensor<2304x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x768xf32>, tensor<768x768xf32>, tensor<768x768xf32>) -> tensor<2304x50xf32>
    %355 = "mhlo.fusion"(%354) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [0:768, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %356 = mhlo.bitcast %355 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %357 = mhlo.bitcast %arg182 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %358 = "mhlo.fusion"(%354) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [768:1536, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %359 = mhlo.bitcast %358 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %360 = "mhlo.fusion"(%348, %356, %357, %359) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x64xf32>, %arg202: tensor<12x64x50xf32>, %arg203: tensor<12x64xf32>, %arg204: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg201, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<12x64x50xf32>
      %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
      %384 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x64x50xf32>
      %385 = stablehlo.multiply %383, %384 : tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %387 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %388 = stablehlo.add %387, %arg204 : tensor<12x64x50xf32>
      %389 = stablehlo.multiply %388, %384 : tensor<12x64x50xf32>
      %390 = stablehlo.dot_general %386, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x50x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
      mhlo.return %390 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x64xf32>, tensor<12x64x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<12x50x50xf32>
    %361 = "mhlo.fusion"(%360) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>):
      %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %383 = stablehlo.compare EQ, %arg201, %382 : (tensor<12x50x50xf32>, tensor<12x50x50xf32>) -> tensor<12x50x50xi1>
      %384 = stablehlo.not %383 : tensor<12x50x50xi1>
      %c = stablehlo.constant dense<false> : tensor<i1>
      %385 = stablehlo.reduce(%384 init: %c) applies stablehlo.or across dimensions = [2] : (tensor<12x50x50xi1>, tensor<i1>) -> tensor<12x50xi1>
      %386 = stablehlo.not %385 : tensor<12x50xi1>
      %387 = stablehlo.broadcast_in_dim %386, dims = [0, 1] : (tensor<12x50xi1>) -> tensor<12x50x50xi1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<12x50x50xf32>
      %389 = stablehlo.reduce(%arg201 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %391 = stablehlo.subtract %arg201, %390 : tensor<12x50x50xf32>
      %392 = stablehlo.exponential %391 : tensor<12x50x50xf32>
      %393 = stablehlo.reduce(%392 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<12x50x50xf32>, tensor<f32>) -> tensor<12x50xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [0, 1] : (tensor<12x50xf32>) -> tensor<12x50x50xf32>
      %395 = stablehlo.divide %392, %394 : tensor<12x50x50xf32>
      %396 = stablehlo.select %387, %388, %395 : tensor<12x50x50xi1>, tensor<12x50x50xf32>
      mhlo.return %396 : tensor<12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<12x50x50xf32>) -> tensor<12x50x50xf32>
    %362 = mhlo.bitcast %arg184 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<12x64xf32>
    %363 = "mhlo.fusion"(%354) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2304x50xf32>):
      %382 = stablehlo.slice %arg201 [1536:2304, 0:50] : (tensor<2304x50xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<2304x50xf32>) -> tensor<768x50xf32>
    %364 = mhlo.bitcast %363 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<12x64x50xf32>
    %365 = "mhlo.fusion"(%361, %362, %364) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<12x50x50xf32>, %arg202: tensor<12x64xf32>, %arg203: tensor<12x64x50xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1] : (tensor<12x64xf32>) -> tensor<12x64x50xf32>
      %383 = stablehlo.add %382, %arg203 : tensor<12x64x50xf32>
      %384 = stablehlo.transpose %383, dims = [0, 2, 1] : (tensor<12x64x50xf32>) -> tensor<12x50x64xf32>
      %385 = stablehlo.dot_general %384, %arg201, batching_dims = [0] x [0], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[12,64,50]{1,2,0}"} : (tensor<12x50x64xf32>, tensor<12x50x50xf32>) -> tensor<12x64x50xf32>
      %386 = stablehlo.transpose %385, dims = [2, 1, 0] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
      mhlo.return %386 : tensor<50x64x12xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[50,64,12]{1,2,0}"} : (tensor<12x50x50xf32>, tensor<12x64xf32>, tensor<12x64x50xf32>) -> tensor<50x64x12xf32>
    %366 = mhlo.bitcast %365 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 2, 0]> : tensor<3xindex>} : (tensor<50x64x12xf32>) -> tensor<50x768xf32>
    %367 = "mhlo.fusion"(%366, %arg187) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768x768xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x768xf32>, tensor<50x768xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<768x768xf32>) -> tensor<768x50xf32>
    %368 = "mhlo.fusion"(%347, %367, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      mhlo.return %385 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<1x768x50xf32>
    %369 = "mhlo.fusion"(%368) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %383 = stablehlo.reduce(%382 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %383 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %370 = "mhlo.fusion"(%368, %369) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %387 = stablehlo.subtract %arg201, %386 : tensor<1x768x50xf32>
      mhlo.return %387 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %371 = "mhlo.fusion"(%370) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>):
      %382 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x50xf32>
      %383 = mhlo.bitcast %382 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x50xf32>) -> tensor<768x50xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %384 = stablehlo.reduce(%383 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x50xf32>, tensor<f32>) -> tensor<50xf32>
      mhlo.return %384 : tensor<50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>) -> tensor<50xf32>
    %372 = "mhlo.fusion"(%370, %371) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<50xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<50xf32>) -> tensor<1x50xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x50xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %385 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50xf32>
      %386 = stablehlo.add %384, %385 : tensor<1x50xf32>
      %387 = stablehlo.rsqrt %386 : tensor<1x50xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xf32>) -> tensor<50xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [2] : (tensor<50xf32>) -> tensor<1x768x50xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x768x50xf32>
      mhlo.return %390 : tensor<1x768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<50xf32>) -> tensor<1x768x50xf32>
    %373 = mhlo.bitcast %372 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<1x768x50xf32>) -> tensor<50x768xf32>
    %374 = "mhlo.fusion"(%373, %arg195, %arg196, %arg191) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %383 = stablehlo.multiply %arg201, %382 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %384 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : (tensor<768xf32>) -> tensor<50x768xf32>
      %385 = stablehlo.add %383, %384 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[50,768]{0,1}"} : tensor<50x768xf32>
      %386 = stablehlo.dot_general %arg204, %385, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<3072x768xf32>, tensor<50x768xf32>) -> tensor<3072x50xf32>
      mhlo.return %386 : tensor<3072x50xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,50]{0,1}"} : (tensor<50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x50xf32>
    %375 = "mhlo.fusion"(%374, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3072x50xf32>, %arg202: tensor<3072xf32>):
      %382 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<50x3072xf32>
      %383 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x50xf32>) -> tensor<50x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<50x3072xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<50x3072xf32>) -> tensor<1x50x3072xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x50x3072xf32>
      %388 = stablehlo.multiply %385, %387 : tensor<1x50x3072xf32>
      %389 = stablehlo.negate %388 : tensor<1x50x3072xf32>
      %390 = stablehlo.exponential %389 : tensor<1x50x3072xf32>
      %391 = stablehlo.add %386, %390 : tensor<1x50x3072xf32>
      %392 = stablehlo.divide %386, %391 : tensor<1x50x3072xf32>
      %393 = stablehlo.multiply %385, %392 : tensor<1x50x3072xf32>
      mhlo.return %393 : tensor<1x50x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x50xf32>, tensor<3072xf32>) -> tensor<1x50x3072xf32>
    %376 = mhlo.bitcast %375 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x3072xf32>) -> tensor<50x3072xf32>
    %377 = "mhlo.fusion"(%376, %arg193) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<50x3072xf32>, %arg202: tensor<768x3072xf32>):
      %382 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<50x3072xf32>) -> tensor<768x50xf32>
      mhlo.return %382 : tensor<768x50xf32>
    }) {output_operand_aliasing = []} : (tensor<50x3072xf32>, tensor<768x3072xf32>) -> tensor<768x50xf32>
    %378 = "mhlo.fusion"(%368, %377, %arg194) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x50xf32>, %arg202: tensor<768x50xf32>, %arg203: tensor<768xf32>):
      %382 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x50xf32>
      %383 = stablehlo.add %382, %arg202 : tensor<768x50xf32>
      %384 = mhlo.bitcast %383 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x50xf32>) -> tensor<1x768x50xf32>
      %385 = stablehlo.add %arg201, %384 : tensor<1x768x50xf32>
      %386 = stablehlo.slice %385 [0:1, 0:768, 1:50] : (tensor<1x768x50xf32>) -> tensor<1x768x49xf32>
      %387 = mhlo.bitcast %386 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x768x49xf32>) -> tensor<768x49xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.reduce(%387 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<768x49xf32>, tensor<f32>) -> tensor<768xf32>
      mhlo.return %388 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x50xf32>, tensor<768x50xf32>, tensor<768xf32>) -> tensor<768xf32>
    %379 = "mhlo.fusion"(%arg197, %378) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<2x768xf32>, %arg202: tensor<768xf32>):
      %382 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %384 = stablehlo.multiply %382, %383 : tensor<1x768xf32>
      %385 = mhlo.bitcast %384 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %386 = stablehlo.broadcast_in_dim %385, dims = [1] : (tensor<768xf32>) -> tensor<2x768xf32>
      %387 = stablehlo.multiply %386, %arg201 : tensor<2x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %388 = stablehlo.reduce(%387 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<2x768xf32>, tensor<f32>) -> tensor<2xf32>
      mhlo.return %388 : tensor<2xf32>
    }) {output_operand_aliasing = []} : (tensor<2x768xf32>, tensor<768xf32>) -> tensor<2xf32>
    %380 = "mhlo.fusion"(%arg198, %379) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<2xf32>, %arg202: tensor<2xf32>):
      %382 = stablehlo.add %arg201, %arg202 : tensor<2xf32>
      mhlo.return %382 : tensor<2xf32>
    }) {output_operand_aliasing = []} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %381 = mhlo.bitcast %380 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<2xf32>) -> tensor<1x2xf32>
    return %381 : tensor<1x2xf32>
  }
}
