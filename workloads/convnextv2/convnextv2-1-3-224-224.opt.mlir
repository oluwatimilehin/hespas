module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.37_computation(%arg0: tensor<3136x96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<384x96xf32>) -> tensor<3136x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    return %4 : tensor<3136x384xf32>
  }
  func.func private @gemm_fusion_dot_general.38_computation(%arg0: tensor<3136x384xf32>, %arg1: tensor<96x384xf32>) -> tensor<3136x96xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    return %0 : tensor<3136x96xf32>
  }
  func.func private @gemm_fusion_dot_general.39_computation(%arg0: tensor<3136x96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<384x96xf32>) -> tensor<3136x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    return %4 : tensor<3136x384xf32>
  }
  func.func private @gemm_fusion_dot_general.40_computation(%arg0: tensor<3136x384xf32>, %arg1: tensor<96x384xf32>) -> tensor<3136x96xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    return %0 : tensor<3136x96xf32>
  }
  func.func private @gemm_fusion_dot_general.41_computation(%arg0: tensor<3136x96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<384x96xf32>) -> tensor<3136x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    return %4 : tensor<3136x384xf32>
  }
  func.func private @gemm_fusion_dot_general.42_computation(%arg0: tensor<3136x384xf32>, %arg1: tensor<96x384xf32>) -> tensor<3136x96xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    return %0 : tensor<3136x96xf32>
  }
  func.func private @gemm_fusion_dot_general.43_computation(%arg0: tensor<784x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<768x192xf32>) -> tensor<784x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    return %4 : tensor<784x768xf32>
  }
  func.func private @gemm_fusion_dot_general.44_computation(%arg0: tensor<784x768xf32>, %arg1: tensor<192x768xf32>) -> tensor<784x192xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    return %0 : tensor<784x192xf32>
  }
  func.func private @gemm_fusion_dot_general.45_computation(%arg0: tensor<784x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<768x192xf32>) -> tensor<784x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    return %4 : tensor<784x768xf32>
  }
  func.func private @gemm_fusion_dot_general.46_computation(%arg0: tensor<784x768xf32>, %arg1: tensor<192x768xf32>) -> tensor<784x192xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    return %0 : tensor<784x192xf32>
  }
  func.func private @gemm_fusion_dot_general.47_computation(%arg0: tensor<784x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<768x192xf32>) -> tensor<784x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    return %4 : tensor<784x768xf32>
  }
  func.func private @gemm_fusion_dot_general.48_computation(%arg0: tensor<784x768xf32>, %arg1: tensor<192x768xf32>) -> tensor<784x192xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    return %0 : tensor<784x192xf32>
  }
  func.func private @gemm_fusion_dot_general.49_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.50_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.51_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.52_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.53_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.54_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.55_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.56_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.57_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.58_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.59_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.60_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.61_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.62_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.63_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.64_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.65_computation(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1536x384xf32>) -> tensor<196x1536xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    return %4 : tensor<196x1536xf32>
  }
  func.func private @gemm_fusion_dot_general.66_computation(%arg0: tensor<196x1536xf32>, %arg1: tensor<384x1536xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    return %0 : tensor<196x384xf32>
  }
  func.func private @gemm_fusion_dot_general.67_computation(%arg0: tensor<49x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<3072x768xf32>, tensor<49x768xf32>) -> tensor<3072x49xf32>
    return %4 : tensor<3072x49xf32>
  }
  func.func private @gemm_fusion_dot_general.68_computation(%arg0: tensor<49x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<49x3072xf32>) -> tensor<768x49xf32>
    return %0 : tensor<768x49xf32>
  }
  func.func private @gemm_fusion_dot_general.69_computation(%arg0: tensor<49x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<3072x768xf32>, tensor<49x768xf32>) -> tensor<3072x49xf32>
    return %4 : tensor<3072x49xf32>
  }
  func.func private @gemm_fusion_dot_general.70_computation(%arg0: tensor<49x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<49x3072xf32>) -> tensor<768x49xf32>
    return %0 : tensor<768x49xf32>
  }
  func.func private @gemm_fusion_dot_general.71_computation(%arg0: tensor<49x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<3072x768xf32>) -> tensor<3072x49xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
    %1 = stablehlo.multiply %arg0, %0 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
    %3 = stablehlo.add %1, %2 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<3072x768xf32>, tensor<49x768xf32>) -> tensor<3072x49xf32>
    return %4 : tensor<3072x49xf32>
  }
  func.func private @gemm_fusion_dot_general.72_computation(%arg0: tensor<49x3072xf32>, %arg1: tensor<768x3072xf32>) -> tensor<768x49xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<49x3072xf32>) -> tensor<768x49xf32>
    return %0 : tensor<768x49xf32>
  }
  func.func private @region_0.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768x49xf32>, %arg2: tensor<768xf32>) -> tensor<768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x49xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x49xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x49xf32>) -> tensor<1x768x7x7xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x49xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<768x49xf32>, tensor<f32>) -> tensor<768xf32>
    return %5 : tensor<768xf32>
  }
  func.func private @region_0.1.clone.82(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<56x56xf32>) -> tensor<56x56xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x96x56x56xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x96x56x56xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %8 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.81(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<3136x96xf32>, %arg2: tensor<96xf32>) -> (tensor<56x56xf32>, tensor<1x96x56x56xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<96xf32>) -> tensor<96x3136xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<96x3136xf32>
    %2 = stablehlo.add %0, %1 : tensor<96x3136xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<96x3136xf32>) -> tensor<1x96x56x56xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x96x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %6, %4 : tensor<56x56xf32>, tensor<1x96x56x56xf32>
  }
  func.func private @region_0.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<f32>, %arg1: tensor<768xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x768xf32>
    %3 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %4 = stablehlo.multiply %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<1x768xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x768xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
    return %10 : tensor<f32>
  }
  func.func private @region_0.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<768xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
    return %4 : tensor<f32>
  }
  func.func private @region_0.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<1x3072xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3072xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_multiply_reduce(%arg0: tensor<3072x49xf32>, %arg1: tensor<3072xf32>) -> (tensor<1x3072xf32>, tensor<1x7x7x3072xf32>) {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x49xf32>) -> tensor<49x3072xf32>
    %3 = stablehlo.add %1, %2 : tensor<49x3072xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<49x3072xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %6 = stablehlo.negate %3 : tensor<49x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<49x3072xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %10 = stablehlo.abs %9 : tensor<1x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %12 = stablehlo.compare LT, %10, %11 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %14 = stablehlo.multiply %9, %9 : tensor<1x7x7x3072xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x7x7x3072xf32>
    %18 = stablehlo.multiply %17, %14 : tensor<1x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x7x7x3072xf32>
    %21 = stablehlo.multiply %20, %14 : tensor<1x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %23 = stablehlo.add %21, %22 : tensor<1x7x7x3072xf32>
    %24 = stablehlo.multiply %23, %14 : tensor<1x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %26 = stablehlo.add %24, %25 : tensor<1x7x7x3072xf32>
    %27 = stablehlo.multiply %26, %14 : tensor<1x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %29 = stablehlo.add %27, %28 : tensor<1x7x7x3072xf32>
    %30 = stablehlo.multiply %29, %14 : tensor<1x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %32 = stablehlo.add %30, %31 : tensor<1x7x7x3072xf32>
    %33 = stablehlo.multiply %9, %32 : tensor<1x7x7x3072xf32>
    %34 = stablehlo.subtract %11, %33 : tensor<1x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %36 = stablehlo.compare LT, %9, %35 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %38 = stablehlo.negate %14 : tensor<1x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %40 = stablehlo.compare LT, %38, %39 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %41 = stablehlo.exponential %38 : tensor<1x7x7x3072xf32>
    %42 = stablehlo.divide %11, %10 : tensor<1x7x7x3072xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x7x7x3072xf32>
    %44 = stablehlo.compare LT, %10, %37 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %46 = stablehlo.divide %11, %14 : tensor<1x7x7x3072xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %48 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %49 = stablehlo.add %47, %48 : tensor<1x7x7x3072xf32>
    %50 = stablehlo.multiply %49, %46 : tensor<1x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x7x7x3072xf32>
    %53 = stablehlo.multiply %52, %46 : tensor<1x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %55 = stablehlo.add %53, %54 : tensor<1x7x7x3072xf32>
    %56 = stablehlo.multiply %55, %46 : tensor<1x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %58 = stablehlo.add %56, %57 : tensor<1x7x7x3072xf32>
    %59 = stablehlo.multiply %58, %46 : tensor<1x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %61 = stablehlo.add %59, %60 : tensor<1x7x7x3072xf32>
    %62 = stablehlo.multiply %61, %46 : tensor<1x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %64 = stablehlo.add %62, %63 : tensor<1x7x7x3072xf32>
    %65 = stablehlo.multiply %64, %46 : tensor<1x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %67 = stablehlo.add %65, %66 : tensor<1x7x7x3072xf32>
    %68 = stablehlo.multiply %67, %46 : tensor<1x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %70 = stablehlo.add %68, %69 : tensor<1x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %72 = stablehlo.multiply %71, %46 : tensor<1x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %74 = stablehlo.add %72, %73 : tensor<1x7x7x3072xf32>
    %75 = stablehlo.multiply %74, %46 : tensor<1x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %77 = stablehlo.add %75, %76 : tensor<1x7x7x3072xf32>
    %78 = stablehlo.multiply %77, %46 : tensor<1x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %79 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %80 = stablehlo.add %78, %79 : tensor<1x7x7x3072xf32>
    %81 = stablehlo.multiply %80, %46 : tensor<1x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %82 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %83 = stablehlo.add %81, %82 : tensor<1x7x7x3072xf32>
    %84 = stablehlo.multiply %83, %46 : tensor<1x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %86 = stablehlo.add %84, %85 : tensor<1x7x7x3072xf32>
    %87 = stablehlo.multiply %86, %46 : tensor<1x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %89 = stablehlo.add %87, %88 : tensor<1x7x7x3072xf32>
    %90 = stablehlo.multiply %89, %46 : tensor<1x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %92 = stablehlo.add %90, %91 : tensor<1x7x7x3072xf32>
    %93 = stablehlo.select %44, %70, %92 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %94 = stablehlo.multiply %43, %93 : tensor<1x7x7x3072xf32>
    %95 = stablehlo.select %40, %35, %94 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %96 = stablehlo.subtract %37, %95 : tensor<1x7x7x3072xf32>
    %97 = stablehlo.select %36, %96, %95 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %98 = stablehlo.select %12, %34, %97 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %99 = stablehlo.multiply %5, %98 : tensor<1x7x7x3072xf32>
    %100 = stablehlo.multiply %99, %99 : tensor<1x7x7x3072xf32>
    %101 = mhlo.bitcast %100 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    %102 = stablehlo.reduce(%101 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<49x3072xf32>, tensor<f32>) -> tensor<3072xf32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %104 = stablehlo.sqrt %103 : tensor<1x3072xf32>
    return %104, %99 : tensor<1x3072xf32>, tensor<1x7x7x3072xf32>
  }
  func.func private @region_0.1.clone.80(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<1x768x7x7xf32>) -> tensor<7x7xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x7x7xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
    return %2 : tensor<7x7xf32>
  }
  func.func private @region_0.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<1x768x7x7xf32>) -> tensor<7x7xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
    return %1 : tensor<7x7xf32>
  }
  func.func private @region_0.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<1x3072xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3072xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_multiply_reduce.1(%arg0: tensor<3072x49xf32>, %arg1: tensor<3072xf32>) -> (tensor<1x3072xf32>, tensor<1x7x7x3072xf32>) {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x49xf32>) -> tensor<49x3072xf32>
    %3 = stablehlo.add %1, %2 : tensor<49x3072xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<49x3072xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %6 = stablehlo.negate %3 : tensor<49x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<49x3072xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %10 = stablehlo.abs %9 : tensor<1x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %12 = stablehlo.compare LT, %10, %11 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %14 = stablehlo.multiply %9, %9 : tensor<1x7x7x3072xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x7x7x3072xf32>
    %18 = stablehlo.multiply %17, %14 : tensor<1x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x7x7x3072xf32>
    %21 = stablehlo.multiply %20, %14 : tensor<1x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %23 = stablehlo.add %21, %22 : tensor<1x7x7x3072xf32>
    %24 = stablehlo.multiply %23, %14 : tensor<1x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %26 = stablehlo.add %24, %25 : tensor<1x7x7x3072xf32>
    %27 = stablehlo.multiply %26, %14 : tensor<1x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %29 = stablehlo.add %27, %28 : tensor<1x7x7x3072xf32>
    %30 = stablehlo.multiply %29, %14 : tensor<1x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %32 = stablehlo.add %30, %31 : tensor<1x7x7x3072xf32>
    %33 = stablehlo.multiply %9, %32 : tensor<1x7x7x3072xf32>
    %34 = stablehlo.subtract %11, %33 : tensor<1x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %36 = stablehlo.compare LT, %9, %35 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %38 = stablehlo.negate %14 : tensor<1x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %40 = stablehlo.compare LT, %38, %39 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %41 = stablehlo.exponential %38 : tensor<1x7x7x3072xf32>
    %42 = stablehlo.divide %11, %10 : tensor<1x7x7x3072xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x7x7x3072xf32>
    %44 = stablehlo.compare LT, %10, %37 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %46 = stablehlo.divide %11, %14 : tensor<1x7x7x3072xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %48 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %49 = stablehlo.add %47, %48 : tensor<1x7x7x3072xf32>
    %50 = stablehlo.multiply %49, %46 : tensor<1x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x7x7x3072xf32>
    %53 = stablehlo.multiply %52, %46 : tensor<1x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %55 = stablehlo.add %53, %54 : tensor<1x7x7x3072xf32>
    %56 = stablehlo.multiply %55, %46 : tensor<1x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %58 = stablehlo.add %56, %57 : tensor<1x7x7x3072xf32>
    %59 = stablehlo.multiply %58, %46 : tensor<1x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %61 = stablehlo.add %59, %60 : tensor<1x7x7x3072xf32>
    %62 = stablehlo.multiply %61, %46 : tensor<1x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %64 = stablehlo.add %62, %63 : tensor<1x7x7x3072xf32>
    %65 = stablehlo.multiply %64, %46 : tensor<1x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %67 = stablehlo.add %65, %66 : tensor<1x7x7x3072xf32>
    %68 = stablehlo.multiply %67, %46 : tensor<1x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %70 = stablehlo.add %68, %69 : tensor<1x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %72 = stablehlo.multiply %71, %46 : tensor<1x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %74 = stablehlo.add %72, %73 : tensor<1x7x7x3072xf32>
    %75 = stablehlo.multiply %74, %46 : tensor<1x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %77 = stablehlo.add %75, %76 : tensor<1x7x7x3072xf32>
    %78 = stablehlo.multiply %77, %46 : tensor<1x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %79 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %80 = stablehlo.add %78, %79 : tensor<1x7x7x3072xf32>
    %81 = stablehlo.multiply %80, %46 : tensor<1x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %82 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %83 = stablehlo.add %81, %82 : tensor<1x7x7x3072xf32>
    %84 = stablehlo.multiply %83, %46 : tensor<1x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %86 = stablehlo.add %84, %85 : tensor<1x7x7x3072xf32>
    %87 = stablehlo.multiply %86, %46 : tensor<1x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %89 = stablehlo.add %87, %88 : tensor<1x7x7x3072xf32>
    %90 = stablehlo.multiply %89, %46 : tensor<1x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %92 = stablehlo.add %90, %91 : tensor<1x7x7x3072xf32>
    %93 = stablehlo.select %44, %70, %92 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %94 = stablehlo.multiply %43, %93 : tensor<1x7x7x3072xf32>
    %95 = stablehlo.select %40, %35, %94 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %96 = stablehlo.subtract %37, %95 : tensor<1x7x7x3072xf32>
    %97 = stablehlo.select %36, %96, %95 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %98 = stablehlo.select %12, %34, %97 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %99 = stablehlo.multiply %5, %98 : tensor<1x7x7x3072xf32>
    %100 = stablehlo.multiply %99, %99 : tensor<1x7x7x3072xf32>
    %101 = mhlo.bitcast %100 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    %102 = stablehlo.reduce(%101 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<49x3072xf32>, tensor<f32>) -> tensor<3072xf32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %104 = stablehlo.sqrt %103 : tensor<1x3072xf32>
    return %104, %99 : tensor<1x3072xf32>, tensor<1x7x7x3072xf32>
  }
  func.func private @region_0.1.clone.79(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<1x768x7x7xf32>) -> tensor<7x7xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x7x7xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
    return %2 : tensor<7x7xf32>
  }
  func.func private @region_0.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<1x768x7x7xf32>) -> tensor<7x7xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
    return %1 : tensor<7x7xf32>
  }
  func.func private @region_0.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<1x3072xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3072xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_multiply_reduce.2(%arg0: tensor<3072x49xf32>, %arg1: tensor<3072xf32>) -> (tensor<1x3072xf32>, tensor<1x7x7x3072xf32>) {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x49xf32>) -> tensor<49x3072xf32>
    %3 = stablehlo.add %1, %2 : tensor<49x3072xf32>
    %4 = stablehlo.multiply %0, %3 : tensor<49x3072xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %6 = stablehlo.negate %3 : tensor<49x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<49x3072xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %10 = stablehlo.abs %9 : tensor<1x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %12 = stablehlo.compare LT, %10, %11 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %14 = stablehlo.multiply %9, %9 : tensor<1x7x7x3072xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x7x7x3072xf32>
    %18 = stablehlo.multiply %17, %14 : tensor<1x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x7x7x3072xf32>
    %21 = stablehlo.multiply %20, %14 : tensor<1x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %23 = stablehlo.add %21, %22 : tensor<1x7x7x3072xf32>
    %24 = stablehlo.multiply %23, %14 : tensor<1x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %25 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %26 = stablehlo.add %24, %25 : tensor<1x7x7x3072xf32>
    %27 = stablehlo.multiply %26, %14 : tensor<1x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %29 = stablehlo.add %27, %28 : tensor<1x7x7x3072xf32>
    %30 = stablehlo.multiply %29, %14 : tensor<1x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %32 = stablehlo.add %30, %31 : tensor<1x7x7x3072xf32>
    %33 = stablehlo.multiply %9, %32 : tensor<1x7x7x3072xf32>
    %34 = stablehlo.subtract %11, %33 : tensor<1x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %36 = stablehlo.compare LT, %9, %35 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %38 = stablehlo.negate %14 : tensor<1x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %40 = stablehlo.compare LT, %38, %39 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %41 = stablehlo.exponential %38 : tensor<1x7x7x3072xf32>
    %42 = stablehlo.divide %11, %10 : tensor<1x7x7x3072xf32>
    %43 = stablehlo.multiply %41, %42 : tensor<1x7x7x3072xf32>
    %44 = stablehlo.compare LT, %10, %37 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %46 = stablehlo.divide %11, %14 : tensor<1x7x7x3072xf32>
    %47 = stablehlo.multiply %45, %46 : tensor<1x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %48 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %49 = stablehlo.add %47, %48 : tensor<1x7x7x3072xf32>
    %50 = stablehlo.multiply %49, %46 : tensor<1x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %52 = stablehlo.add %50, %51 : tensor<1x7x7x3072xf32>
    %53 = stablehlo.multiply %52, %46 : tensor<1x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %55 = stablehlo.add %53, %54 : tensor<1x7x7x3072xf32>
    %56 = stablehlo.multiply %55, %46 : tensor<1x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %58 = stablehlo.add %56, %57 : tensor<1x7x7x3072xf32>
    %59 = stablehlo.multiply %58, %46 : tensor<1x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %61 = stablehlo.add %59, %60 : tensor<1x7x7x3072xf32>
    %62 = stablehlo.multiply %61, %46 : tensor<1x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %63 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %64 = stablehlo.add %62, %63 : tensor<1x7x7x3072xf32>
    %65 = stablehlo.multiply %64, %46 : tensor<1x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %67 = stablehlo.add %65, %66 : tensor<1x7x7x3072xf32>
    %68 = stablehlo.multiply %67, %46 : tensor<1x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %69 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %70 = stablehlo.add %68, %69 : tensor<1x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %72 = stablehlo.multiply %71, %46 : tensor<1x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %74 = stablehlo.add %72, %73 : tensor<1x7x7x3072xf32>
    %75 = stablehlo.multiply %74, %46 : tensor<1x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %77 = stablehlo.add %75, %76 : tensor<1x7x7x3072xf32>
    %78 = stablehlo.multiply %77, %46 : tensor<1x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %79 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %80 = stablehlo.add %78, %79 : tensor<1x7x7x3072xf32>
    %81 = stablehlo.multiply %80, %46 : tensor<1x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %82 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %83 = stablehlo.add %81, %82 : tensor<1x7x7x3072xf32>
    %84 = stablehlo.multiply %83, %46 : tensor<1x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %86 = stablehlo.add %84, %85 : tensor<1x7x7x3072xf32>
    %87 = stablehlo.multiply %86, %46 : tensor<1x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %89 = stablehlo.add %87, %88 : tensor<1x7x7x3072xf32>
    %90 = stablehlo.multiply %89, %46 : tensor<1x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %92 = stablehlo.add %90, %91 : tensor<1x7x7x3072xf32>
    %93 = stablehlo.select %44, %70, %92 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %94 = stablehlo.multiply %43, %93 : tensor<1x7x7x3072xf32>
    %95 = stablehlo.select %40, %35, %94 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %96 = stablehlo.subtract %37, %95 : tensor<1x7x7x3072xf32>
    %97 = stablehlo.select %36, %96, %95 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %98 = stablehlo.select %12, %34, %97 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %99 = stablehlo.multiply %5, %98 : tensor<1x7x7x3072xf32>
    %100 = stablehlo.multiply %99, %99 : tensor<1x7x7x3072xf32>
    %101 = mhlo.bitcast %100 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    %102 = stablehlo.reduce(%101 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<49x3072xf32>, tensor<f32>) -> tensor<3072xf32>
    %103 = mhlo.bitcast %102 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %104 = stablehlo.sqrt %103 : tensor<1x3072xf32>
    return %104, %99 : tensor<1x3072xf32>, tensor<1x7x7x3072xf32>
  }
  func.func private @region_0.1.clone.78(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<1x768x7x7xf32>) -> tensor<7x7xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x768x7x7xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
    return %2 : tensor<7x7xf32>
  }
  func.func private @region_0.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<1x768x7x7xf32>) -> tensor<7x7xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
    return %1 : tensor<7x7xf32>
  }
  func.func private @region_0.1.clone.77(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %1 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.76(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.75(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.74(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.73(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.72(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.71(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<1x1536xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x1536xf32>
    return %4 : tensor<1x1536xf32>
  }
  func.func private @region_0.1.clone.68(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<1x384x14x14xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x384x14x14xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %2 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<384xf32>) -> tensor<14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
    return %3 : tensor<14x14xf32>
  }
  func.func private @region_0.1.clone.67(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<28x28xf32>) -> tensor<28x28xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x192x28x28xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x192x28x28xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %8 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_add_reduce.1(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<784x192xf32>, %arg2: tensor<192xf32>) -> (tensor<28x28xf32>, tensor<1x192x28x28xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<192xf32>) -> tensor<192x784xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<192x784xf32>
    %2 = stablehlo.add %0, %1 : tensor<192x784xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x784xf32>) -> tensor<1x192x28x28xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x192x28x28xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %6, %4 : tensor<28x28xf32>, tensor<1x192x28x28xf32>
  }
  func.func private @region_0.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.52(%arg0: tensor<1x768xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.53(%arg0: tensor<1x28x28x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x28x28x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<784x768xf32>, tensor<f32>) -> tensor<768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x768xf32>
    return %4 : tensor<1x768xf32>
  }
  func.func private @region_0.1.clone.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.54(%arg0: tensor<1x192x28x28xf32>) -> tensor<28x28xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x192x28x28xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %2 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.55(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<192xf32>) -> tensor<28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x192x28x28xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %3 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.56(%arg0: tensor<1x768xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.57(%arg0: tensor<1x28x28x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x28x28x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<784x768xf32>, tensor<f32>) -> tensor<768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x768xf32>
    return %4 : tensor<1x768xf32>
  }
  func.func private @region_0.1.clone.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.58(%arg0: tensor<1x192x28x28xf32>) -> tensor<28x28xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x192x28x28xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %2 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.59(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<192xf32>) -> tensor<28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x192x28x28xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %3 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.60(%arg0: tensor<1x768xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.61(%arg0: tensor<1x28x28x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x28x28x768xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<784x768xf32>, tensor<f32>) -> tensor<768xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x768xf32>
    return %4 : tensor<1x768xf32>
  }
  func.func private @region_0.1.clone.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.62(%arg0: tensor<1x192x28x28xf32>) -> tensor<28x28xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x192x28x28xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %2 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.63(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<192xf32>) -> tensor<28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x192x28x28xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
    return %3 : tensor<28x28xf32>
  }
  func.func private @region_0.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.64(%arg0: tensor<1x384xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.65(%arg0: tensor<1x56x56x384xf32>) -> tensor<1x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x56x56x384xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3136x384xf32>, tensor<f32>) -> tensor<384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<384xf32>) -> tensor<1x384xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x384xf32>
    return %4 : tensor<1x384xf32>
  }
  func.func private @region_0.1.clone.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.66(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>, %arg2: tensor<56x56xf32>) -> tensor<56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x96x56x56xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x96x56x56xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %10 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.67(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>) -> tensor<56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %3 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.68(%arg0: tensor<1x384xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.69(%arg0: tensor<1x56x56x384xf32>) -> tensor<1x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x56x56x384xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3136x384xf32>, tensor<f32>) -> tensor<384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<384xf32>) -> tensor<1x384xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x384xf32>
    return %4 : tensor<1x384xf32>
  }
  func.func private @region_0.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.70(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>, %arg2: tensor<56x56xf32>) -> tensor<56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x96x56x56xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x96x56x56xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %10 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.71(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>) -> tensor<56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %3 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.72(%arg0: tensor<1x384xf32>) -> tensor<f32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384xf32>, tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.73(%arg0: tensor<1x56x56x384xf32>) -> tensor<1x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x56x56x384xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3136x384xf32>, tensor<f32>) -> tensor<384xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<384xf32>) -> tensor<1x384xf32>
    %4 = stablehlo.sqrt %3 : tensor<1x384xf32>
    return %4 : tensor<1x384xf32>
  }
  func.func private @region_0.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.74(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>, %arg2: tensor<56x56xf32>) -> tensor<56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x96x56x56xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<1x96x56x56xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%9 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %10 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.75(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>) -> tensor<56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %3 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.76(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<56x56xf32>) -> tensor<56x56xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x96x56x56xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x96x56x56xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %8 : tensor<56x56xf32>
  }
  func.func private @region_0.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.77(%arg0: tensor<1x96x56x56xf32>) -> tensor<56x56xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
    return %1 : tensor<56x56xf32>
  }
  func.func private @fused_add(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768x49xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x49xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x49xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x49xf32>) -> tensor<1x768x7x7xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x7x7xf32>
    return %3 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768x49xf32>, %arg2: tensor<768xf32>) -> tensor<1x768x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<768xf32>) -> tensor<768x49xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<768x49xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x49xf32>) -> tensor<1x768x7x7xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x768x7x7xf32>
    return %3 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
    %2 = stablehlo.add %0, %1 : tensor<384x196xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x384x14x14xf32>
    return %4 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<784x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x192x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<192xf32>) -> tensor<192x784xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<192x784xf32>
    %2 = stablehlo.add %0, %1 : tensor<192x784xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x784xf32>) -> tensor<1x192x28x28xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x192x28x28xf32>
    return %4 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<784x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x192x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<192xf32>) -> tensor<192x784xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<192x784xf32>
    %2 = stablehlo.add %0, %1 : tensor<192x784xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x784xf32>) -> tensor<1x192x28x28xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x192x28x28xf32>
    return %4 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<3136x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x96x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<96xf32>) -> tensor<96x3136xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<96x3136xf32>
    %2 = stablehlo.add %0, %1 : tensor<96x3136xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<96x3136xf32>) -> tensor<1x96x56x56xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x96x56x56xf32>
    return %4 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<3136x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x96x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<96xf32>) -> tensor<96x3136xf32>
    %1 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<96x3136xf32>
    %2 = stablehlo.add %0, %1 : tensor<96x3136xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<96x3136xf32>) -> tensor<1x96x56x56xf32>
    %4 = stablehlo.add %arg0, %3 : tensor<1x96x56x56xf32>
    return %4 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<768xf32>) -> tensor<1x768xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x768xf32>
    %3 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %4 = stablehlo.multiply %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %7 = stablehlo.subtract %2, %6 : tensor<1x768xf32>
    %8 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %9 = stablehlo.multiply %8, %cst_0 : tensor<1xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<1xf32>
    %10 = stablehlo.add %9, %cst_1 : tensor<1xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %14 = stablehlo.multiply %7, %13 : tensor<1x768xf32>
    %15 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x768xf32>
    %17 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x768xf32>
    return %18 : tensor<1x768xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<196x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<196x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<196x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<196x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<1x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<1x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<1x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x14x14x1536xf32>
    return %98 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<784x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x28x28x768xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<784x768xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<784x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
    %5 = stablehlo.negate %2 : tensor<784x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<784x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
    %9 = stablehlo.abs %8 : tensor<1x28x28x768xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x28x28x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x28x28x768xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x28x28x768xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x28x28x768xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x28x28x768xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x28x28x768xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x28x28x768xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x28x28x768xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x28x28x768xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x28x28x768xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x28x28x768xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x28x28x768xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x28x28x768xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x28x28x768xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x28x28x768xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %37 = stablehlo.negate %13 : tensor<1x28x28x768xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %40 = stablehlo.exponential %37 : tensor<1x28x28x768xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x28x28x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x28x28x768xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x28x28x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x28x28x768xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x28x28x768xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x28x28x768xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x28x28x768xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x28x28x768xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x28x28x768xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x28x28x768xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x28x28x768xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x28x28x768xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x28x28x768xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x28x28x768xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x28x28x768xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x28x28x768xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x28x28x768xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x28x28x768xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x28x28x768xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x28x28x768xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x28x28x768xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x28x28x768xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x28x28x768xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x28x28x768xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x28x28x768xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x28x28x768xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x28x28x768xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x28x28x768xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x28x28x768xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x28x28x768xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x28x28x768xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x28x28x768xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x28x28x768xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x28x28x768xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x28x28x768xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x28x28x768xf32>
    return %98 : tensor<1x28x28x768xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<784x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x28x28x768xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<784x768xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<784x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
    %5 = stablehlo.negate %2 : tensor<784x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<784x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
    %9 = stablehlo.abs %8 : tensor<1x28x28x768xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x28x28x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x28x28x768xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x28x28x768xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x28x28x768xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x28x28x768xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x28x28x768xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x28x28x768xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x28x28x768xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x28x28x768xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x28x28x768xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x28x28x768xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x28x28x768xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x28x28x768xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x28x28x768xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x28x28x768xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %37 = stablehlo.negate %13 : tensor<1x28x28x768xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %40 = stablehlo.exponential %37 : tensor<1x28x28x768xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x28x28x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x28x28x768xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x28x28x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x28x28x768xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x28x28x768xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x28x28x768xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x28x28x768xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x28x28x768xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x28x28x768xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x28x28x768xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x28x28x768xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x28x28x768xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x28x28x768xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x28x28x768xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x28x28x768xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x28x28x768xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x28x28x768xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x28x28x768xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x28x28x768xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x28x28x768xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x28x28x768xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x28x28x768xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x28x28x768xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x28x28x768xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x28x28x768xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x28x28x768xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x28x28x768xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x28x28x768xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x28x28x768xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x28x28x768xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x28x28x768xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x28x28x768xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x28x28x768xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x28x28x768xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x28x28x768xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x28x28x768xf32>
    return %98 : tensor<1x28x28x768xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<784x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x28x28x768xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<784x768xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<784x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
    %5 = stablehlo.negate %2 : tensor<784x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<784x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
    %9 = stablehlo.abs %8 : tensor<1x28x28x768xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x28x28x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x28x28x768xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x28x28x768xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x28x28x768xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x28x28x768xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x28x28x768xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x28x28x768xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x28x28x768xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x28x28x768xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x28x28x768xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x28x28x768xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x28x28x768xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x28x28x768xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x28x28x768xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x28x28x768xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %37 = stablehlo.negate %13 : tensor<1x28x28x768xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %40 = stablehlo.exponential %37 : tensor<1x28x28x768xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x28x28x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x28x28x768xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x28x28x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x28x28x768xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x28x28x768xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x28x28x768xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x28x28x768xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x28x28x768xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x28x28x768xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x28x28x768xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x28x28x768xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x28x28x768xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x28x28x768xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x28x28x768xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x28x28x768xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x28x28x768xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x28x28x768xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x28x28x768xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x28x28x768xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x28x28x768xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x28x28x768xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x28x28x768xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x28x28x768xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x28x28x768xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x28x28x768xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x28x28x768xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x28x28x768xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x28x28x768xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x28x28x768xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x28x28x768xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x28x28x768xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x28x28x768xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x28x28x768xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x28x28x768xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x28x28x768xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x28x28x768xf32>
    return %98 : tensor<1x28x28x768xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<3136x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x56x56x384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3136x384xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3136x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
    %5 = stablehlo.negate %2 : tensor<3136x384xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3136x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
    %9 = stablehlo.abs %8 : tensor<1x56x56x384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x56x56x384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x56x56x384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x56x56x384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x56x56x384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x56x56x384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x56x56x384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x56x56x384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x56x56x384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x56x56x384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x56x56x384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x56x56x384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x56x56x384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x56x56x384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x56x56x384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x56x56x384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %37 = stablehlo.negate %13 : tensor<1x56x56x384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %40 = stablehlo.exponential %37 : tensor<1x56x56x384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x56x56x384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x56x56x384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x56x56x384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x56x56x384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x56x56x384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x56x56x384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x56x56x384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x56x56x384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x56x56x384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x56x56x384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x56x56x384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x56x56x384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x56x56x384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x56x56x384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x56x56x384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x56x56x384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x56x56x384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x56x56x384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x56x56x384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x56x56x384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x56x56x384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x56x56x384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x56x56x384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x56x56x384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x56x56x384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x56x56x384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x56x56x384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x56x56x384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x56x56x384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x56x56x384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x56x56x384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x56x56x384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x56x56x384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x56x56x384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x56x56x384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x56x56x384xf32>
    return %98 : tensor<1x56x56x384xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<3136x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x56x56x384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3136x384xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3136x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
    %5 = stablehlo.negate %2 : tensor<3136x384xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3136x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
    %9 = stablehlo.abs %8 : tensor<1x56x56x384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x56x56x384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x56x56x384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x56x56x384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x56x56x384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x56x56x384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x56x56x384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x56x56x384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x56x56x384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x56x56x384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x56x56x384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x56x56x384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x56x56x384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x56x56x384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x56x56x384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x56x56x384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %37 = stablehlo.negate %13 : tensor<1x56x56x384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %40 = stablehlo.exponential %37 : tensor<1x56x56x384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x56x56x384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x56x56x384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x56x56x384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x56x56x384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x56x56x384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x56x56x384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x56x56x384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x56x56x384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x56x56x384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x56x56x384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x56x56x384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x56x56x384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x56x56x384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x56x56x384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x56x56x384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x56x56x384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x56x56x384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x56x56x384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x56x56x384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x56x56x384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x56x56x384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x56x56x384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x56x56x384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x56x56x384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x56x56x384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x56x56x384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x56x56x384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x56x56x384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x56x56x384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x56x56x384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x56x56x384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x56x56x384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x56x56x384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x56x56x384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x56x56x384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x56x56x384xf32>
    return %98 : tensor<1x56x56x384xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<3136x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x56x56x384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<3136x384xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<3136x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
    %5 = stablehlo.negate %2 : tensor<3136x384xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<3136x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
    %9 = stablehlo.abs %8 : tensor<1x56x56x384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x56x56x384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x56x56x384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x56x56x384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x56x56x384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x56x56x384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x56x56x384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x56x56x384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x56x56x384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x56x56x384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x56x56x384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x56x56x384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x56x56x384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x56x56x384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x56x56x384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x56x56x384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %37 = stablehlo.negate %13 : tensor<1x56x56x384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %40 = stablehlo.exponential %37 : tensor<1x56x56x384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x56x56x384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x56x56x384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x56x56x384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x56x56x384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x56x56x384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x56x56x384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x56x56x384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x56x56x384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x56x56x384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x56x56x384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x56x56x384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x56x56x384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x56x56x384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x56x56x384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x56x56x384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x56x56x384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x56x56x384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x56x56x384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x56x56x384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x56x56x384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x56x56x384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x56x56x384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x56x56x384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x56x56x384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x56x56x384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x56x56x384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x56x56x384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x56x56x384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x56x56x384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x56x56x384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x56x56x384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x56x56x384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x56x56x384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x56x56x384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x56x56x384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x56x56x384xf32>
    return %98 : tensor<1x56x56x384xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<1x1x1x384xf32>, %arg3: tensor<1x384xf32>, %arg4: tensor<f32>) -> tensor<1x56x56x384xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x384xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x56x56x384xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x56x56x384xf32>
    %12 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x56x56x384xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x56x56x384xf32>
    return %15 : tensor<1x56x56x384xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<1x1x1x384xf32>, %arg3: tensor<1x384xf32>, %arg4: tensor<f32>) -> tensor<1x56x56x384xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x384xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x56x56x384xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x56x56x384xf32>
    %12 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x56x56x384xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x56x56x384xf32>
    return %15 : tensor<1x56x56x384xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<1x1x1x384xf32>, %arg3: tensor<1x384xf32>, %arg4: tensor<f32>) -> tensor<1x56x56x384xf32> {
    %0 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x384xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x56x56x384xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x56x56x384xf32>
    %12 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x56x56x384xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x56x56x384xf32>
    return %15 : tensor<1x56x56x384xf32>
  }
  func.func private @region_0.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.78(%arg0: tensor<1000x768xf32>, %arg1: tensor<1x768xf32>) -> tensor<1000xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<768xf32>) -> tensor<1000x768xf32>
    %2 = stablehlo.multiply %1, %arg0 : tensor<1000x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1000x768xf32>, tensor<f32>) -> tensor<1000xf32>
    return %3 : tensor<1000xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<1x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<1x1x1x768xf32>, %arg3: tensor<1x768xf32>, %arg4: tensor<f32>) -> tensor<1x28x28x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x28x28x768xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x28x28x768xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x28x28x768xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x28x28x768xf32>
    return %15 : tensor<1x28x28x768xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<1x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<1x1x1x768xf32>, %arg3: tensor<1x768xf32>, %arg4: tensor<f32>) -> tensor<1x28x28x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x28x28x768xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x28x28x768xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x28x28x768xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x28x28x768xf32>
    return %15 : tensor<1x28x28x768xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<1x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<1x1x1x768xf32>, %arg3: tensor<1x768xf32>, %arg4: tensor<f32>) -> tensor<1x28x28x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x28x28x768xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x28x28x768xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x28x28x768xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x28x28x768xf32>
    return %15 : tensor<1x28x28x768xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x96x56x56xf32>, %arg3: tensor<56x56xf32>, %arg4: tensor<56x56xf32>) -> tensor<1x96x56x56xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x96x56x56xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x56x56xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x56x56xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x96x56x56xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x96x56x56xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x96x56x56xf32>
    return %17 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>, %arg2: tensor<56x56xf32>, %arg3: tensor<56x56xf32>) -> tensor<1x96x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x96x56x56xf32>
    %8 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %9 = stablehlo.multiply %8, %3 : tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x56x56xf32>
    %12 = stablehlo.rsqrt %11 : tensor<1x56x56xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<1x96x56x56xf32>
    return %15 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>, %arg2: tensor<56x56xf32>, %arg3: tensor<56x56xf32>) -> tensor<1x96x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x96x56x56xf32>
    %8 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %9 = stablehlo.multiply %8, %3 : tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x56x56xf32>
    %12 = stablehlo.rsqrt %11 : tensor<1x56x56xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<1x96x56x56xf32>
    return %15 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<1x96x56x56xf32>, %arg1: tensor<96xf32>, %arg2: tensor<56x56xf32>, %arg3: tensor<56x56xf32>) -> tensor<1x96x56x56xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x96x56x56xf32>
    %2 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x56x56xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x96x56x56xf32>
    %8 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %9 = stablehlo.multiply %8, %3 : tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x56x56xf32>
    %12 = stablehlo.rsqrt %11 : tensor<1x56x56xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %15 = stablehlo.multiply %7, %14 : tensor<1x96x56x56xf32>
    return %15 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x96x56x56xf32>, %arg3: tensor<56x56xf32>, %arg4: tensor<56x56xf32>) -> tensor<1x96x56x56xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x96x56x56xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x56x56xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x56x56xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x96x56x56xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x96x56x56xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x96x56x56xf32>
    return %17 : tensor<1x96x56x56xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.26(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.27(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.28(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.29(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.30(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.31(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_add.32(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1536xf32>, %arg4: tensor<f32>) -> tensor<1x14x14x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x14x14x1536xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x14x14x1536xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x14x14x1536xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x14x14x1536xf32>
    return %15 : tensor<1x14x14x1536xf32>
  }
  func.func private @fused_subtract(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<192xf32>, %arg2: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x192x28x28xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x28x28xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x192x28x28xf32>
    return %7 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x192x28x28xf32>
    return %8 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_subtract.1(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<192xf32>, %arg2: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x192x28x28xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x28x28xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x192x28x28xf32>
    return %7 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x192x28x28xf32>
    return %8 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_subtract.2(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<192xf32>, %arg2: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x192x28x28xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x28x28xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x192x28x28xf32>
    return %7 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<1x192x28x28xf32>, %arg1: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x192x28x28xf32>
    return %8 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_add.33(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x192x28x28xf32>, %arg3: tensor<28x28xf32>, %arg4: tensor<28x28xf32>) -> tensor<1x192x28x28xf32> {
    %0 = mhlo.bitcast %arg4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<1x192x28x28xf32>
    %6 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
    %7 = stablehlo.multiply %6, %1 : tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x28x28xf32>
    %10 = stablehlo.rsqrt %9 : tensor<1x28x28xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %13 = stablehlo.multiply %5, %12 : tensor<1x192x28x28xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<1x192x28x28xf32>
    %16 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %17 = stablehlo.add %15, %16 : tensor<1x192x28x28xf32>
    return %17 : tensor<1x192x28x28xf32>
  }
  func.func private @fused_add.34(%arg0: tensor<1x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<1x1x1x3072xf32>, %arg3: tensor<1x3072xf32>, %arg4: tensor<f32>) -> tensor<1x7x7x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x3072xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x7x7x3072xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x7x7x3072xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x7x7x3072xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x7x7x3072xf32>
    return %15 : tensor<1x7x7x3072xf32>
  }
  func.func private @fused_add.35(%arg0: tensor<1x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<1x1x1x3072xf32>, %arg3: tensor<1x3072xf32>, %arg4: tensor<f32>) -> tensor<1x7x7x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x3072xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x7x7x3072xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x7x7x3072xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x7x7x3072xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x7x7x3072xf32>
    return %15 : tensor<1x7x7x3072xf32>
  }
  func.func private @fused_add.36(%arg0: tensor<1x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<1x1x1x3072xf32>, %arg3: tensor<1x3072xf32>, %arg4: tensor<f32>) -> tensor<1x7x7x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %2 = mhlo.bitcast %arg4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1xf32>
    %3 = stablehlo.multiply %2, %cst : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
    %4 = stablehlo.add %3, %cst_0 : tensor<1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x3072xf32>
    %7 = stablehlo.divide %arg3, %6 : tensor<1x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<1x7x7x3072xf32>
    %11 = stablehlo.multiply %1, %10 : tensor<1x7x7x3072xf32>
    %12 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x7x7x3072xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<1x7x7x3072xf32>
    return %15 : tensor<1x7x7x3072xf32>
  }
  func.func private @fused_subtract.3(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.4(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.5(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.6(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.7(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.8(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.9(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.10(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.11(%arg0: tensor<14x14xf32>, %arg1: tensor<1x384x14x14xf32>, %arg2: tensor<384xf32>) -> tensor<1x384x14x14xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x384x14x14xf32>
    %2 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x14x14xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %7 = stablehlo.subtract %1, %6 : tensor<1x384x14x14xf32>
    return %7 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x384x14x14xf32>
    return %8 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.12(%arg0: tensor<1x384x14x14xf32>, %arg1: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x384x14x14xf32>
    return %5 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_add.37(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x384x14x14xf32>, %arg3: tensor<14x14xf32>) -> tensor<1x384x14x14xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<1x384x14x14xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x384x14x14xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %12 = stablehlo.add %10, %11 : tensor<1x384x14x14xf32>
    return %12 : tensor<1x384x14x14xf32>
  }
  func.func private @fused_subtract.13(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<7x7xf32>) -> tensor<1x768x7x7xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x7x7xf32>
    return %5 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<7x7xf32>) -> tensor<1x768x7x7xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x7x7xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x7x7xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x7x7xf32>
    return %8 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_subtract.14(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<7x7xf32>) -> tensor<1x768x7x7xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x7x7xf32>
    return %5 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<7x7xf32>) -> tensor<1x768x7x7xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x7x7xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x7x7xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x7x7xf32>
    return %8 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_subtract.15(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<7x7xf32>) -> tensor<1x768x7x7xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x768x7x7xf32>
    return %5 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<7x7xf32>) -> tensor<1x768x7x7xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x7x7xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x7x7xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<1x768x7x7xf32>
    return %8 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_add.38(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768xf32>) -> tensor<1x768x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x768x7x7xf32>
    return %1 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_add.39(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768xf32>) -> tensor<1x768x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x768x7x7xf32>
    return %1 : tensor<1x768x7x7xf32>
  }
  func.func private @fused_add.40(%arg0: tensor<1x768x7x7xf32>, %arg1: tensor<768xf32>) -> tensor<1x768x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x768x7x7xf32>
    return %1 : tensor<1x768x7x7xf32>
  }
  func.func private @wrapped_add_computation(%arg0: tensor<1000xf32>, %arg1: tensor<1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func.func @main(%arg0: tensor<96x3x4x4xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<96xf32>, %arg4: tensor<96x1x7x7xf32>, %arg5: tensor<96xf32>, %arg6: tensor<96xf32>, %arg7: tensor<96xf32>, %arg8: tensor<384x96xf32>, %arg9: tensor<384xf32>, %arg10: tensor<1x1x1x384xf32>, %arg11: tensor<1x1x1x384xf32>, %arg12: tensor<96x384xf32>, %arg13: tensor<96xf32>, %arg14: tensor<96x1x7x7xf32>, %arg15: tensor<96xf32>, %arg16: tensor<96xf32>, %arg17: tensor<96xf32>, %arg18: tensor<384x96xf32>, %arg19: tensor<384xf32>, %arg20: tensor<1x1x1x384xf32>, %arg21: tensor<1x1x1x384xf32>, %arg22: tensor<96x384xf32>, %arg23: tensor<96xf32>, %arg24: tensor<96x1x7x7xf32>, %arg25: tensor<96xf32>, %arg26: tensor<96xf32>, %arg27: tensor<96xf32>, %arg28: tensor<384x96xf32>, %arg29: tensor<384xf32>, %arg30: tensor<1x1x1x384xf32>, %arg31: tensor<1x1x1x384xf32>, %arg32: tensor<96x384xf32>, %arg33: tensor<96xf32>, %arg34: tensor<96xf32>, %arg35: tensor<96xf32>, %arg36: tensor<192x96x2x2xf32>, %arg37: tensor<192xf32>, %arg38: tensor<192x1x7x7xf32>, %arg39: tensor<192xf32>, %arg40: tensor<192xf32>, %arg41: tensor<192xf32>, %arg42: tensor<768x192xf32>, %arg43: tensor<768xf32>, %arg44: tensor<1x1x1x768xf32>, %arg45: tensor<1x1x1x768xf32>, %arg46: tensor<192x768xf32>, %arg47: tensor<192xf32>, %arg48: tensor<192x1x7x7xf32>, %arg49: tensor<192xf32>, %arg50: tensor<192xf32>, %arg51: tensor<192xf32>, %arg52: tensor<768x192xf32>, %arg53: tensor<768xf32>, %arg54: tensor<1x1x1x768xf32>, %arg55: tensor<1x1x1x768xf32>, %arg56: tensor<192x768xf32>, %arg57: tensor<192xf32>, %arg58: tensor<192x1x7x7xf32>, %arg59: tensor<192xf32>, %arg60: tensor<192xf32>, %arg61: tensor<192xf32>, %arg62: tensor<768x192xf32>, %arg63: tensor<768xf32>, %arg64: tensor<1x1x1x768xf32>, %arg65: tensor<1x1x1x768xf32>, %arg66: tensor<192x768xf32>, %arg67: tensor<192xf32>, %arg68: tensor<192xf32>, %arg69: tensor<192xf32>, %arg70: tensor<384x192x2x2xf32>, %arg71: tensor<384xf32>, %arg72: tensor<384x1x7x7xf32>, %arg73: tensor<384xf32>, %arg74: tensor<384xf32>, %arg75: tensor<384xf32>, %arg76: tensor<1536x384xf32>, %arg77: tensor<1536xf32>, %arg78: tensor<1x1x1x1536xf32>, %arg79: tensor<1x1x1x1536xf32>, %arg80: tensor<384x1536xf32>, %arg81: tensor<384xf32>, %arg82: tensor<384x1x7x7xf32>, %arg83: tensor<384xf32>, %arg84: tensor<384xf32>, %arg85: tensor<384xf32>, %arg86: tensor<1536x384xf32>, %arg87: tensor<1536xf32>, %arg88: tensor<1x1x1x1536xf32>, %arg89: tensor<1x1x1x1536xf32>, %arg90: tensor<384x1536xf32>, %arg91: tensor<384xf32>, %arg92: tensor<384x1x7x7xf32>, %arg93: tensor<384xf32>, %arg94: tensor<384xf32>, %arg95: tensor<384xf32>, %arg96: tensor<1536x384xf32>, %arg97: tensor<1536xf32>, %arg98: tensor<1x1x1x1536xf32>, %arg99: tensor<1x1x1x1536xf32>, %arg100: tensor<384x1536xf32>, %arg101: tensor<384xf32>, %arg102: tensor<384x1x7x7xf32>, %arg103: tensor<384xf32>, %arg104: tensor<384xf32>, %arg105: tensor<384xf32>, %arg106: tensor<1536x384xf32>, %arg107: tensor<1536xf32>, %arg108: tensor<1x1x1x1536xf32>, %arg109: tensor<1x1x1x1536xf32>, %arg110: tensor<384x1536xf32>, %arg111: tensor<384xf32>, %arg112: tensor<384x1x7x7xf32>, %arg113: tensor<384xf32>, %arg114: tensor<384xf32>, %arg115: tensor<384xf32>, %arg116: tensor<1536x384xf32>, %arg117: tensor<1536xf32>, %arg118: tensor<1x1x1x1536xf32>, %arg119: tensor<1x1x1x1536xf32>, %arg120: tensor<384x1536xf32>, %arg121: tensor<384xf32>, %arg122: tensor<384x1x7x7xf32>, %arg123: tensor<384xf32>, %arg124: tensor<384xf32>, %arg125: tensor<384xf32>, %arg126: tensor<1536x384xf32>, %arg127: tensor<1536xf32>, %arg128: tensor<1x1x1x1536xf32>, %arg129: tensor<1x1x1x1536xf32>, %arg130: tensor<384x1536xf32>, %arg131: tensor<384xf32>, %arg132: tensor<384x1x7x7xf32>, %arg133: tensor<384xf32>, %arg134: tensor<384xf32>, %arg135: tensor<384xf32>, %arg136: tensor<1536x384xf32>, %arg137: tensor<1536xf32>, %arg138: tensor<1x1x1x1536xf32>, %arg139: tensor<1x1x1x1536xf32>, %arg140: tensor<384x1536xf32>, %arg141: tensor<384xf32>, %arg142: tensor<384x1x7x7xf32>, %arg143: tensor<384xf32>, %arg144: tensor<384xf32>, %arg145: tensor<384xf32>, %arg146: tensor<1536x384xf32>, %arg147: tensor<1536xf32>, %arg148: tensor<1x1x1x1536xf32>, %arg149: tensor<1x1x1x1536xf32>, %arg150: tensor<384x1536xf32>, %arg151: tensor<384xf32>, %arg152: tensor<384x1x7x7xf32>, %arg153: tensor<384xf32>, %arg154: tensor<384xf32>, %arg155: tensor<384xf32>, %arg156: tensor<1536x384xf32>, %arg157: tensor<1536xf32>, %arg158: tensor<1x1x1x1536xf32>, %arg159: tensor<1x1x1x1536xf32>, %arg160: tensor<384x1536xf32>, %arg161: tensor<384xf32>, %arg162: tensor<384xf32>, %arg163: tensor<384xf32>, %arg164: tensor<768x384x2x2xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x1x7x7xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<3072x768xf32>, %arg171: tensor<3072xf32>, %arg172: tensor<1x1x1x3072xf32>, %arg173: tensor<1x1x1x3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768x1x7x7xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<3072x768xf32>, %arg181: tensor<3072xf32>, %arg182: tensor<1x1x1x3072xf32>, %arg183: tensor<1x1x1x3072xf32>, %arg184: tensor<768x3072xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x1x7x7xf32>, %arg187: tensor<768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<3072x768xf32>, %arg191: tensor<3072xf32>, %arg192: tensor<1x1x1x3072xf32>, %arg193: tensor<1x1x1x3072xf32>, %arg194: tensor<768x3072xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%arg200, %arg0, %arg1) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,96,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x3x224x224xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>) -> tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>) -> tensor<1x96x56x56xf32>
    %2 = "mhlo.fusion"(%1) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %293 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>) -> tensor<56x56xf32>
    %3 = "mhlo.fusion"(%1, %2) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<56x56xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x56x56xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x96x56x56xf32>
      %298 = stablehlo.multiply %297, %297 : tensor<1x96x56x56xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %300 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<56x56xf32>) -> tensor<56x56xf32>
    %4 = "mhlo.fusion"(%arg3, %arg2, %1, %3, %2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x96x56x56xf32>, %arg204: tensor<56x56xf32>, %arg205: tensor<56x56xf32>):
      %292 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x56x56xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %297 = stablehlo.subtract %arg203, %296 : tensor<1x96x56x56xf32>
      %298 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %299 = stablehlo.multiply %298, %293 : tensor<1x56x56xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %301 = stablehlo.add %299, %300 : tensor<1x56x56xf32>
      %302 = stablehlo.rsqrt %301 : tensor<1x56x56xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %305 = stablehlo.multiply %297, %304 : tensor<1x96x56x56xf32>
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<1x96x56x56xf32>
      %308 = stablehlo.broadcast_in_dim %arg201, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %309 = stablehlo.add %307, %308 : tensor<1x96x56x56xf32>
      mhlo.return %309 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x96x56x56xf32>, tensor<56x56xf32>, tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %5 = stablehlo.custom_call @__cudnn$convForward(%4, %arg4) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,96,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>) -> tensor<1x96x56x56xf32>
    %7 = "mhlo.fusion"(%6, %arg5) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %295 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>) -> tensor<56x56xf32>
    %8 = "mhlo.fusion"(%6, %arg5, %7) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>, %arg203: tensor<56x56xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x96x56x56xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x96x56x56xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%301 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %302 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>, tensor<56x56xf32>) -> tensor<56x56xf32>
    %9 = "mhlo.fusion"(%6, %arg5, %8, %7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>, %arg203: tensor<56x56xf32>, %arg204: tensor<56x56xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x96x56x56xf32>
      %300 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %301 = stablehlo.multiply %300, %295 : tensor<1x56x56xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %303 = stablehlo.add %301, %302 : tensor<1x56x56xf32>
      %304 = stablehlo.rsqrt %303 : tensor<1x56x56xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %307 = stablehlo.multiply %299, %306 : tensor<1x96x56x56xf32>
      mhlo.return %307 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>, tensor<56x56xf32>, tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<1x96x56x56xf32>) -> tensor<3136x96xf32>
    %11 = "mhlo.fusion"(%10, %arg6, %arg7, %arg8) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3136x96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<96xf32>, %arg204: tensor<384x96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
      mhlo.return %296 : tensor<3136x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    %12 = "mhlo.fusion"(%11, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<384xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<3136x384xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<3136x384xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %297 = stablehlo.negate %294 : tensor<3136x384xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<3136x384xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %301 = stablehlo.abs %300 : tensor<1x56x56x384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x56x56x384xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x56x56x384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x56x56x384xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x56x56x384xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x56x56x384xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x56x56x384xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x56x56x384xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x56x56x384xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x56x56x384xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %329 = stablehlo.negate %305 : tensor<1x56x56x384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %332 = stablehlo.exponential %329 : tensor<1x56x56x384xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x56x56x384xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x56x56x384xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x56x56x384xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x56x56x384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x56x56x384xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x56x56x384xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x56x56x384xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x56x56x384xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x56x56x384xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x56x56x384xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x56x56x384xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x56x56x384xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x56x56x384xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x56x56x384xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x56x56x384xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x56x56x384xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x56x56x384xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x56x56x384xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x56x56x384xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x56x56x384xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x56x56x384xf32>
      mhlo.return %390 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x384xf32>, tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %13 = "mhlo.fusion"(%12) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x56x56x384xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3136x384xf32>, tensor<f32>) -> tensor<384xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<384xf32>) -> tensor<1x384xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x384xf32>
      mhlo.return %296 : tensor<1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>) -> tensor<1x384xf32>
    %14 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x384xf32>) -> tensor<f32>
    %15 = "mhlo.fusion"(%12, %arg11, %arg10, %13, %14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<1x1x1x384xf32>, %arg204: tensor<1x384xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x384xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x384xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x56x56x384xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x56x56x384xf32>
      %304 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x56x56x384xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x56x56x384xf32>
      mhlo.return %307 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>, tensor<1x384xf32>, tensor<f32>) -> tensor<1x56x56x384xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    %17 = "mhlo.fusion"(%16, %arg12) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<96x384xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
      mhlo.return %292 : tensor<3136x96xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    %18 = "mhlo.fusion"(%4, %17, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<3136x96xf32>, %arg203: tensor<96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<96xf32>) -> tensor<96x3136xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<96x3136xf32>
      %294 = stablehlo.add %292, %293 : tensor<96x3136xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<96x3136xf32>) -> tensor<1x96x56x56xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x96x56x56xf32>
      mhlo.return %296 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<3136x96xf32>, tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %19 = stablehlo.custom_call @__cudnn$convForward(%18, %arg14) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,96,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>
    %20 = stablehlo.get_tuple_element %19[0] : (tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>) -> tensor<1x96x56x56xf32>
    %21 = "mhlo.fusion"(%20, %arg15) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %295 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>) -> tensor<56x56xf32>
    %22 = "mhlo.fusion"(%20, %arg15, %21) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>, %arg203: tensor<56x56xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x96x56x56xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x96x56x56xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%301 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %302 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>, tensor<56x56xf32>) -> tensor<56x56xf32>
    %23 = "mhlo.fusion"(%20, %arg15, %22, %21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>, %arg203: tensor<56x56xf32>, %arg204: tensor<56x56xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x96x56x56xf32>
      %300 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %301 = stablehlo.multiply %300, %295 : tensor<1x56x56xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %303 = stablehlo.add %301, %302 : tensor<1x56x56xf32>
      %304 = stablehlo.rsqrt %303 : tensor<1x56x56xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %307 = stablehlo.multiply %299, %306 : tensor<1x96x56x56xf32>
      mhlo.return %307 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>, tensor<56x56xf32>, tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %24 = mhlo.bitcast %23 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<1x96x56x56xf32>) -> tensor<3136x96xf32>
    %25 = "mhlo.fusion"(%24, %arg16, %arg17, %arg18) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3136x96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<96xf32>, %arg204: tensor<384x96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
      mhlo.return %296 : tensor<3136x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    %26 = "mhlo.fusion"(%25, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<384xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<3136x384xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<3136x384xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %297 = stablehlo.negate %294 : tensor<3136x384xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<3136x384xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %301 = stablehlo.abs %300 : tensor<1x56x56x384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x56x56x384xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x56x56x384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x56x56x384xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x56x56x384xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x56x56x384xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x56x56x384xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x56x56x384xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x56x56x384xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x56x56x384xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %329 = stablehlo.negate %305 : tensor<1x56x56x384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %332 = stablehlo.exponential %329 : tensor<1x56x56x384xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x56x56x384xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x56x56x384xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x56x56x384xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x56x56x384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x56x56x384xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x56x56x384xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x56x56x384xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x56x56x384xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x56x56x384xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x56x56x384xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x56x56x384xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x56x56x384xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x56x56x384xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x56x56x384xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x56x56x384xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x56x56x384xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x56x56x384xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x56x56x384xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x56x56x384xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x56x56x384xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x56x56x384xf32>
      mhlo.return %390 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x384xf32>, tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %27 = "mhlo.fusion"(%26) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x56x56x384xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3136x384xf32>, tensor<f32>) -> tensor<384xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<384xf32>) -> tensor<1x384xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x384xf32>
      mhlo.return %296 : tensor<1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>) -> tensor<1x384xf32>
    %28 = "mhlo.fusion"(%27) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x384xf32>) -> tensor<f32>
    %29 = "mhlo.fusion"(%26, %arg21, %arg20, %27, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<1x1x1x384xf32>, %arg204: tensor<1x384xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x384xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x384xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x56x56x384xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x56x56x384xf32>
      %304 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x56x56x384xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x56x56x384xf32>
      mhlo.return %307 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>, tensor<1x384xf32>, tensor<f32>) -> tensor<1x56x56x384xf32>
    %30 = mhlo.bitcast %29 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    %31 = "mhlo.fusion"(%30, %arg22) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<96x384xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
      mhlo.return %292 : tensor<3136x96xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    %32 = "mhlo.fusion"(%18, %31, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<3136x96xf32>, %arg203: tensor<96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<96xf32>) -> tensor<96x3136xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<96x3136xf32>
      %294 = stablehlo.add %292, %293 : tensor<96x3136xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<96x3136xf32>) -> tensor<1x96x56x56xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x96x56x56xf32>
      mhlo.return %296 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<3136x96xf32>, tensor<96xf32>) -> tensor<1x96x56x56xf32>
    %33 = stablehlo.custom_call @__cudnn$convForward(%32, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,96,56,56]{3,2,1,0}, u8[0]{0})"} : (tensor<1x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>
    %34 = stablehlo.get_tuple_element %33[0] : (tuple<tensor<1x96x56x56xf32>, tensor<0xui8>>) -> tensor<1x96x56x56xf32>
    %35 = "mhlo.fusion"(%34, %arg25) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %295 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>) -> tensor<56x56xf32>
    %36 = "mhlo.fusion"(%34, %arg25, %35) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>, %arg203: tensor<56x56xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x96x56x56xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x96x56x56xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%301 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %302 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>, tensor<56x56xf32>) -> tensor<56x56xf32>
    %37 = "mhlo.fusion"(%34, %arg25, %36, %35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<96xf32>, %arg203: tensor<56x56xf32>, %arg204: tensor<56x56xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x96x56x56xf32>
      %294 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x96x56x56xf32>
      %300 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %301 = stablehlo.multiply %300, %295 : tensor<1x56x56xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %303 = stablehlo.add %301, %302 : tensor<1x56x56xf32>
      %304 = stablehlo.rsqrt %303 : tensor<1x56x56xf32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %307 = stablehlo.multiply %299, %306 : tensor<1x96x56x56xf32>
      mhlo.return %307 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<96xf32>, tensor<56x56xf32>, tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %38 = mhlo.bitcast %37 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<1x96x56x56xf32>) -> tensor<3136x96xf32>
    %39 = "mhlo.fusion"(%38, %arg26, %arg27, %arg28) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3136x96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<96xf32>, %arg204: tensor<384x96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<96xf32>) -> tensor<3136x96xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : tensor<3136x96xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
      mhlo.return %296 : tensor<3136x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    %40 = "mhlo.fusion"(%39, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<384xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<3136x384xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<3136x384xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %297 = stablehlo.negate %294 : tensor<3136x384xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<3136x384xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %301 = stablehlo.abs %300 : tensor<1x56x56x384xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x56x56x384xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x56x56x384xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x56x56x384xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x56x56x384xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x56x56x384xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x56x56x384xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x56x56x384xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x56x56x384xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x56x56x384xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %329 = stablehlo.negate %305 : tensor<1x56x56x384xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %332 = stablehlo.exponential %329 : tensor<1x56x56x384xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x56x56x384xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x56x56x384xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x56x56x384xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x56x56x384xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x56x56x384xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x56x56x384xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x56x56x384xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x56x56x384xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x56x56x384xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x56x56x384xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x56x56x384xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x56x56x384xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x56x56x384xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x56x56x384xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x56x56x384xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x56x56x384xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x56x56x384xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x56x56x384xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x56x56x384xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x56x56x384xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x56x56x384xf32>
      mhlo.return %390 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x384xf32>, tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x56x56x384xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3136x384xf32>, tensor<f32>) -> tensor<384xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<384xf32>) -> tensor<1x384xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x384xf32>
      mhlo.return %296 : tensor<1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>) -> tensor<1x384xf32>
    %42 = "mhlo.fusion"(%41) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x384xf32>) -> tensor<f32>
    %43 = "mhlo.fusion"(%40, %arg31, %arg30, %41, %42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<1x1x1x384xf32>, %arg204: tensor<1x384xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x384xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x384xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<384xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x56x56x384xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x56x56x384xf32>
      %304 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x56x56x384xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x56x56x384xf32>
      mhlo.return %307 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>, tensor<1x384xf32>, tensor<f32>) -> tensor<1x56x56x384xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    %45 = "mhlo.fusion"(%44, %arg32) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<96x384xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
      mhlo.return %292 : tensor<3136x96xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3136,96]{0,1}"} : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    %46:2 = "mhlo.fusion"(%32, %45, %arg33) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<3136x96xf32>, %arg203: tensor<96xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<96xf32>) -> tensor<96x3136xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<96x3136xf32>
      %294 = stablehlo.add %292, %293 : tensor<96x3136xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<96x3136xf32>) -> tensor<1x96x56x56xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x96x56x56xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %298, %296 : tensor<56x56xf32>, tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<3136x96xf32>, tensor<96xf32>) -> (tensor<56x56xf32>, tensor<1x96x56x56xf32>)
    %47 = "mhlo.fusion"(%46#1, %46#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x96x56x56xf32>, %arg202: tensor<56x56xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x56x56xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x96x56x56xf32>
      %298 = stablehlo.multiply %297, %297 : tensor<1x96x56x56xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x96x56x56xf32>) -> tensor<96x56x56xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<96x56x56xf32>, tensor<f32>) -> tensor<56x56xf32>
      mhlo.return %300 : tensor<56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x96x56x56xf32>, tensor<56x56xf32>) -> tensor<56x56xf32>
    %48 = "mhlo.fusion"(%arg35, %arg34, %46#1, %47, %46#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x96x56x56xf32>, %arg204: tensor<56x56xf32>, %arg205: tensor<56x56xf32>):
      %292 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x56x56xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %297 = stablehlo.subtract %arg203, %296 : tensor<1x96x56x56xf32>
      %298 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<56x56xf32>) -> tensor<1x56x56xf32>
      %299 = stablehlo.multiply %298, %293 : tensor<1x56x56xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %301 = stablehlo.add %299, %300 : tensor<1x56x56xf32>
      %302 = stablehlo.rsqrt %301 : tensor<1x56x56xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [2, 3] : (tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
      %305 = stablehlo.multiply %297, %304 : tensor<1x96x56x56xf32>
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<1x96x56x56xf32>
      %308 = stablehlo.broadcast_in_dim %arg201, dims = [1] : (tensor<96xf32>) -> tensor<1x96x56x56xf32>
      %309 = stablehlo.add %307, %308 : tensor<1x96x56x56xf32>
      mhlo.return %309 : tensor<1x96x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x96x56x56xf32>, tensor<56x56xf32>, tensor<56x56xf32>) -> tensor<1x96x56x56xf32>
    %49 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%48, %arg36, %arg37) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x96x56x56xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>) -> tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>
    %50 = stablehlo.get_tuple_element %49[0] : (tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>) -> tensor<1x192x28x28xf32>
    %51 = stablehlo.custom_call @__cudnn$convForward(%50, %arg38) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>
    %52 = stablehlo.get_tuple_element %51[0] : (tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>) -> tensor<1x192x28x28xf32>
    %53 = "mhlo.fusion"(%52, %arg39) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x192x28x28xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %295 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<192xf32>) -> tensor<28x28xf32>
    %54 = "mhlo.fusion"(%52, %arg39, %53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<192xf32>, %arg203: tensor<28x28xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x192x28x28xf32>
      %294 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x28x28xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x192x28x28xf32>
      mhlo.return %299 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<192xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %55 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x192x28x28xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %294 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>) -> tensor<28x28xf32>
    %56 = "mhlo.fusion"(%54, %55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<28x28xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x28x28xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x28x28xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x192x28x28xf32>
      mhlo.return %300 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %57 = mhlo.bitcast %56 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<1x192x28x28xf32>) -> tensor<784x192xf32>
    %58 = "mhlo.fusion"(%57, %arg40, %arg41, %arg42) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<784x192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<192xf32>, %arg204: tensor<768x192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
      mhlo.return %296 : tensor<784x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    %59 = "mhlo.fusion"(%58, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<768xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<784x768xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<784x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %297 = stablehlo.negate %294 : tensor<784x768xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<784x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %301 = stablehlo.abs %300 : tensor<1x28x28x768xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x28x28x768xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x28x28x768xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x28x28x768xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x28x28x768xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x28x28x768xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x28x28x768xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x28x28x768xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x28x28x768xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x28x28x768xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %329 = stablehlo.negate %305 : tensor<1x28x28x768xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %332 = stablehlo.exponential %329 : tensor<1x28x28x768xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x28x28x768xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x28x28x768xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x28x28x768xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x28x28x768xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x28x28x768xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x28x28x768xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x28x28x768xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x28x28x768xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x28x28x768xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x28x28x768xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x28x28x768xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x28x28x768xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x28x28x768xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x28x28x768xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x28x28x768xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x28x28x768xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x28x28x768xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x28x28x768xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x28x28x768xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x28x28x768xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x28x28x768xf32>
      mhlo.return %390 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x768xf32>, tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %60 = "mhlo.fusion"(%59) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x28x28x768xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<784x768xf32>, tensor<f32>) -> tensor<768xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x768xf32>
      mhlo.return %296 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>) -> tensor<1x768xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x768xf32>) -> tensor<f32>
    %62 = "mhlo.fusion"(%59, %arg44, %arg45, %60, %61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<1x1x1x768xf32>, %arg204: tensor<1x768xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x28x28x768xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x28x28x768xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x28x28x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x28x28x768xf32>
      mhlo.return %307 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>, tensor<1x768xf32>, tensor<f32>) -> tensor<1x28x28x768xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    %64 = "mhlo.fusion"(%63, %arg46) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<192x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
      mhlo.return %292 : tensor<784x192xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    %65 = "mhlo.fusion"(%50, %64, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<784x192xf32>, %arg203: tensor<192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<192xf32>) -> tensor<192x784xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<192x784xf32>
      %294 = stablehlo.add %292, %293 : tensor<192x784xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x784xf32>) -> tensor<1x192x28x28xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x192x28x28xf32>
      mhlo.return %296 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<784x192xf32>, tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %66 = stablehlo.custom_call @__cudnn$convForward(%65, %arg48) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>
    %67 = stablehlo.get_tuple_element %66[0] : (tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>) -> tensor<1x192x28x28xf32>
    %68 = "mhlo.fusion"(%67, %arg49) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x192x28x28xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %295 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<192xf32>) -> tensor<28x28xf32>
    %69 = "mhlo.fusion"(%67, %arg49, %68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<192xf32>, %arg203: tensor<28x28xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x192x28x28xf32>
      %294 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x28x28xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x192x28x28xf32>
      mhlo.return %299 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<192xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %70 = "mhlo.fusion"(%69) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x192x28x28xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %294 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>) -> tensor<28x28xf32>
    %71 = "mhlo.fusion"(%69, %70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<28x28xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x28x28xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x28x28xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x192x28x28xf32>
      mhlo.return %300 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %72 = mhlo.bitcast %71 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<1x192x28x28xf32>) -> tensor<784x192xf32>
    %73 = "mhlo.fusion"(%72, %arg50, %arg51, %arg52) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<784x192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<192xf32>, %arg204: tensor<768x192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
      mhlo.return %296 : tensor<784x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    %74 = "mhlo.fusion"(%73, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<768xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<784x768xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<784x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %297 = stablehlo.negate %294 : tensor<784x768xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<784x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %301 = stablehlo.abs %300 : tensor<1x28x28x768xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x28x28x768xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x28x28x768xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x28x28x768xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x28x28x768xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x28x28x768xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x28x28x768xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x28x28x768xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x28x28x768xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x28x28x768xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %329 = stablehlo.negate %305 : tensor<1x28x28x768xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %332 = stablehlo.exponential %329 : tensor<1x28x28x768xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x28x28x768xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x28x28x768xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x28x28x768xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x28x28x768xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x28x28x768xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x28x28x768xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x28x28x768xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x28x28x768xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x28x28x768xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x28x28x768xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x28x28x768xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x28x28x768xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x28x28x768xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x28x28x768xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x28x28x768xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x28x28x768xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x28x28x768xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x28x28x768xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x28x28x768xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x28x28x768xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x28x28x768xf32>
      mhlo.return %390 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x768xf32>, tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %75 = "mhlo.fusion"(%74) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x28x28x768xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<784x768xf32>, tensor<f32>) -> tensor<768xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x768xf32>
      mhlo.return %296 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>) -> tensor<1x768xf32>
    %76 = "mhlo.fusion"(%75) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x768xf32>) -> tensor<f32>
    %77 = "mhlo.fusion"(%74, %arg54, %arg55, %75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<1x1x1x768xf32>, %arg204: tensor<1x768xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x28x28x768xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x28x28x768xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x28x28x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x28x28x768xf32>
      mhlo.return %307 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>, tensor<1x768xf32>, tensor<f32>) -> tensor<1x28x28x768xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    %79 = "mhlo.fusion"(%78, %arg56) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<192x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
      mhlo.return %292 : tensor<784x192xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    %80 = "mhlo.fusion"(%65, %79, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<784x192xf32>, %arg203: tensor<192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<192xf32>) -> tensor<192x784xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<192x784xf32>
      %294 = stablehlo.add %292, %293 : tensor<192x784xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x784xf32>) -> tensor<1x192x28x28xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x192x28x28xf32>
      mhlo.return %296 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<784x192xf32>, tensor<192xf32>) -> tensor<1x192x28x28xf32>
    %81 = stablehlo.custom_call @__cudnn$convForward(%80, %arg58) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,192,28,28]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>
    %82 = stablehlo.get_tuple_element %81[0] : (tuple<tensor<1x192x28x28xf32>, tensor<0xui8>>) -> tensor<1x192x28x28xf32>
    %83 = "mhlo.fusion"(%82, %arg59) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x192x28x28xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %295 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<192xf32>) -> tensor<28x28xf32>
    %84 = "mhlo.fusion"(%82, %arg59, %83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<192xf32>, %arg203: tensor<28x28xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x192x28x28xf32>
      %294 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x28x28xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x192x28x28xf32>
      mhlo.return %299 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<192xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %85 = "mhlo.fusion"(%84) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x192x28x28xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %294 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>) -> tensor<28x28xf32>
    %86 = "mhlo.fusion"(%84, %85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<28x28xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x28x28xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x28x28xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x192x28x28xf32>
      mhlo.return %300 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %87 = mhlo.bitcast %86 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<1x192x28x28xf32>) -> tensor<784x192xf32>
    %88 = "mhlo.fusion"(%87, %arg60, %arg61, %arg62) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<784x192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<192xf32>, %arg204: tensor<768x192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<192xf32>) -> tensor<784x192xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : tensor<784x192xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
      mhlo.return %296 : tensor<784x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    %89 = "mhlo.fusion"(%88, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<768xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<784x768xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<784x768xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %297 = stablehlo.negate %294 : tensor<784x768xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<784x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %301 = stablehlo.abs %300 : tensor<1x28x28x768xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x28x28x768xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x28x28x768xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x28x28x768xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x28x28x768xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x28x28x768xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x28x28x768xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x28x28x768xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x28x28x768xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x28x28x768xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %329 = stablehlo.negate %305 : tensor<1x28x28x768xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %332 = stablehlo.exponential %329 : tensor<1x28x28x768xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x28x28x768xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x28x28x768xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x28x28x768xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x28x28x768xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x28x28x768xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x28x28x768xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x28x28x768xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x28x28x768xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x28x28x768xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x28x28x768xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x28x28x768xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x28x28x768xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x28x28x768xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x28x28x768xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x28x28x768xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x28x28x768xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x28x28x768xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x28x28x768xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x28x28x768xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x28x28x768xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x28x28x768xf32>
      mhlo.return %390 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x768xf32>, tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x28x28x768xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<784x768xf32>, tensor<f32>) -> tensor<768xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x768xf32>
      mhlo.return %296 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>) -> tensor<1x768xf32>
    %91 = "mhlo.fusion"(%90) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x768xf32>) -> tensor<f32>
    %92 = "mhlo.fusion"(%89, %arg64, %arg65, %90, %91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<1x1x1x768xf32>, %arg204: tensor<1x768xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x768xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x28x28x768xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x28x28x768xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x28x28x768xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x28x28x768xf32>
      mhlo.return %307 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>, tensor<1x768xf32>, tensor<f32>) -> tensor<1x28x28x768xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    %94 = "mhlo.fusion"(%93, %arg66) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<192x768xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
      mhlo.return %292 : tensor<784x192xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[784,192]{0,1}"} : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    %95:2 = "mhlo.fusion"(%80, %94, %arg67) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<784x192xf32>, %arg203: tensor<192xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<192xf32>) -> tensor<192x784xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<192x784xf32>
      %294 = stablehlo.add %292, %293 : tensor<192x784xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<192x784xf32>) -> tensor<1x192x28x28xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x192x28x28xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %298 = stablehlo.reduce(%297 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %298, %296 : tensor<28x28xf32>, tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<784x192xf32>, tensor<192xf32>) -> (tensor<28x28xf32>, tensor<1x192x28x28xf32>)
    %96 = "mhlo.fusion"(%95#1, %95#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x192x28x28xf32>, %arg202: tensor<28x28xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x192x28x28xf32>
      %298 = stablehlo.multiply %297, %297 : tensor<1x192x28x28xf32>
      %299 = mhlo.bitcast %298 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x192x28x28xf32>) -> tensor<192x28x28xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<192x28x28xf32>, tensor<f32>) -> tensor<28x28xf32>
      mhlo.return %300 : tensor<28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x192x28x28xf32>, tensor<28x28xf32>) -> tensor<28x28xf32>
    %97 = "mhlo.fusion"(%arg69, %arg68, %95#1, %96, %95#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<1x192x28x28xf32>, %arg204: tensor<28x28xf32>, %arg205: tensor<28x28xf32>):
      %292 = mhlo.bitcast %arg205 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x28x28xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %297 = stablehlo.subtract %arg203, %296 : tensor<1x192x28x28xf32>
      %298 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<28x28xf32>) -> tensor<1x28x28xf32>
      %299 = stablehlo.multiply %298, %293 : tensor<1x28x28xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %301 = stablehlo.add %299, %300 : tensor<1x28x28xf32>
      %302 = stablehlo.rsqrt %301 : tensor<1x28x28xf32>
      %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [2, 3] : (tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
      %305 = stablehlo.multiply %297, %304 : tensor<1x192x28x28xf32>
      %306 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<1x192x28x28xf32>
      %308 = stablehlo.broadcast_in_dim %arg201, dims = [1] : (tensor<192xf32>) -> tensor<1x192x28x28xf32>
      %309 = stablehlo.add %307, %308 : tensor<1x192x28x28xf32>
      mhlo.return %309 : tensor<1x192x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x192x28x28xf32>, tensor<28x28xf32>, tensor<28x28xf32>) -> tensor<1x192x28x28xf32>
    %98 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%97, %arg70, %arg71) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x192x28x28xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %99 = stablehlo.get_tuple_element %98[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %100 = stablehlo.custom_call @__cudnn$convForward(%99, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %101 = stablehlo.get_tuple_element %100[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %102 = "mhlo.fusion"(%101, %arg73) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %103 = "mhlo.fusion"(%102, %101, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %105 = "mhlo.fusion"(%103, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %106 = mhlo.bitcast %105 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %107 = "mhlo.fusion"(%106, %arg74, %arg75, %arg76) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %108 = "mhlo.fusion"(%107, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %110 = "mhlo.fusion"(%109) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %111 = "mhlo.fusion"(%108, %arg78, %arg79, %109, %110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %112 = mhlo.bitcast %111 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %113 = "mhlo.fusion"(%112, %arg80) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %114 = "mhlo.fusion"(%99, %113, %arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %115 = stablehlo.custom_call @__cudnn$convForward(%114, %arg82) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %116 = stablehlo.get_tuple_element %115[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %117 = "mhlo.fusion"(%116, %arg83) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %118 = "mhlo.fusion"(%117, %116, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %120 = "mhlo.fusion"(%118, %119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %121 = mhlo.bitcast %120 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %122 = "mhlo.fusion"(%121, %arg84, %arg85, %arg86) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %123 = "mhlo.fusion"(%122, %arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %124 = "mhlo.fusion"(%123) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %125 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %126 = "mhlo.fusion"(%123, %arg88, %arg89, %124, %125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %127 = mhlo.bitcast %126 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %128 = "mhlo.fusion"(%127, %arg90) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %129 = "mhlo.fusion"(%114, %128, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %130 = stablehlo.custom_call @__cudnn$convForward(%129, %arg92) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %131 = stablehlo.get_tuple_element %130[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %132 = "mhlo.fusion"(%131, %arg93) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %133 = "mhlo.fusion"(%132, %131, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %135 = "mhlo.fusion"(%133, %134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %136 = mhlo.bitcast %135 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %137 = "mhlo.fusion"(%136, %arg94, %arg95, %arg96) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %138 = "mhlo.fusion"(%137, %arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %139 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %140 = "mhlo.fusion"(%139) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %141 = "mhlo.fusion"(%138, %arg98, %arg99, %139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %142 = mhlo.bitcast %141 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %143 = "mhlo.fusion"(%142, %arg100) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %144 = "mhlo.fusion"(%129, %143, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %145 = stablehlo.custom_call @__cudnn$convForward(%144, %arg102) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %146 = stablehlo.get_tuple_element %145[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %147 = "mhlo.fusion"(%146, %arg103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %148 = "mhlo.fusion"(%147, %146, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %150 = "mhlo.fusion"(%148, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %151 = mhlo.bitcast %150 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %152 = "mhlo.fusion"(%151, %arg104, %arg105, %arg106) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %153 = "mhlo.fusion"(%152, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %156 = "mhlo.fusion"(%153, %arg108, %arg109, %154, %155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %157 = mhlo.bitcast %156 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %158 = "mhlo.fusion"(%157, %arg110) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %159 = "mhlo.fusion"(%144, %158, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %160 = stablehlo.custom_call @__cudnn$convForward(%159, %arg112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %161 = stablehlo.get_tuple_element %160[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %162 = "mhlo.fusion"(%161, %arg113) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %163 = "mhlo.fusion"(%162, %161, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %165 = "mhlo.fusion"(%163, %164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %166 = mhlo.bitcast %165 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %167 = "mhlo.fusion"(%166, %arg114, %arg115, %arg116) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %168 = "mhlo.fusion"(%167, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %170 = "mhlo.fusion"(%169) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %171 = "mhlo.fusion"(%168, %arg118, %arg119, %169, %170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %172 = mhlo.bitcast %171 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %173 = "mhlo.fusion"(%172, %arg120) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %174 = "mhlo.fusion"(%159, %173, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %175 = stablehlo.custom_call @__cudnn$convForward(%174, %arg122) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %176 = stablehlo.get_tuple_element %175[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %177 = "mhlo.fusion"(%176, %arg123) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %178 = "mhlo.fusion"(%177, %176, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %180 = "mhlo.fusion"(%178, %179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %181 = mhlo.bitcast %180 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %182 = "mhlo.fusion"(%181, %arg124, %arg125, %arg126) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %183 = "mhlo.fusion"(%182, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %184 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %185 = "mhlo.fusion"(%184) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %186 = "mhlo.fusion"(%183, %arg128, %arg129, %184, %185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %187 = mhlo.bitcast %186 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %188 = "mhlo.fusion"(%187, %arg130) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %189 = "mhlo.fusion"(%174, %188, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %190 = stablehlo.custom_call @__cudnn$convForward(%189, %arg132) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %191 = stablehlo.get_tuple_element %190[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %192 = "mhlo.fusion"(%191, %arg133) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %193 = "mhlo.fusion"(%192, %191, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %194 = "mhlo.fusion"(%193) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %195 = "mhlo.fusion"(%193, %194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %196 = mhlo.bitcast %195 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %197 = "mhlo.fusion"(%196, %arg134, %arg135, %arg136) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %198 = "mhlo.fusion"(%197, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %200 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %201 = "mhlo.fusion"(%198, %arg138, %arg139, %199, %200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %202 = mhlo.bitcast %201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %203 = "mhlo.fusion"(%202, %arg140) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %204 = "mhlo.fusion"(%189, %203, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %205 = stablehlo.custom_call @__cudnn$convForward(%204, %arg142) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %206 = stablehlo.get_tuple_element %205[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %207 = "mhlo.fusion"(%206, %arg143) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %208 = "mhlo.fusion"(%207, %206, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %209 = "mhlo.fusion"(%208) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %210 = "mhlo.fusion"(%208, %209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %211 = mhlo.bitcast %210 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %212 = "mhlo.fusion"(%211, %arg144, %arg145, %arg146) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %213 = "mhlo.fusion"(%212, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %214 = "mhlo.fusion"(%213) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %216 = "mhlo.fusion"(%213, %arg148, %arg149, %214, %215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %217 = mhlo.bitcast %216 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %218 = "mhlo.fusion"(%217, %arg150) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %219 = "mhlo.fusion"(%204, %218, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %220 = stablehlo.custom_call @__cudnn$convForward(%219, %arg152) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,384,14,14]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>
    %221 = stablehlo.get_tuple_element %220[0] : (tuple<tensor<1x384x14x14xf32>, tensor<0xui8>>) -> tensor<1x384x14x14xf32>
    %222 = "mhlo.fusion"(%221, %arg153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %295 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<14x14xf32>
    %223 = "mhlo.fusion"(%222, %221, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<14x14xf32>, %arg202: tensor<1x384x14x14xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %293 = stablehlo.add %arg202, %292 : tensor<1x384x14x14xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.multiply %294, %295 : tensor<1x14x14xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %299 = stablehlo.subtract %293, %298 : tensor<1x384x14x14xf32>
      mhlo.return %299 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<14x14xf32>, tensor<1x384x14x14xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %224 = "mhlo.fusion"(%223) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %225 = "mhlo.fusion"(%223, %224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x384x14x14xf32>
      mhlo.return %300 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<1x384x14x14xf32>) -> tensor<196x384xf32>
    %227 = "mhlo.fusion"(%226, %arg154, %arg155, %arg156) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<384xf32>, %arg204: tensor<1536x384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<384xf32>) -> tensor<196x384xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : tensor<196x384xf32>
      %296 = stablehlo.dot_general %295, %arg204, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
      mhlo.return %296 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %228 = "mhlo.fusion"(%227, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %294 = stablehlo.add %293, %arg201 : tensor<196x1536xf32>
      %295 = stablehlo.multiply %292, %294 : tensor<196x1536xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %297 = stablehlo.negate %294 : tensor<196x1536xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %298 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %299 = stablehlo.multiply %297, %298 : tensor<196x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %301 = stablehlo.abs %300 : tensor<1x14x14x1536xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %303 = stablehlo.compare LT, %301, %302 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %304 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %305 = stablehlo.multiply %300, %300 : tensor<1x14x14x1536xf32>
      %306 = stablehlo.multiply %304, %305 : tensor<1x14x14x1536xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %308 = stablehlo.add %306, %307 : tensor<1x14x14x1536xf32>
      %309 = stablehlo.multiply %308, %305 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %311 = stablehlo.add %309, %310 : tensor<1x14x14x1536xf32>
      %312 = stablehlo.multiply %311, %305 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %314 = stablehlo.add %312, %313 : tensor<1x14x14x1536xf32>
      %315 = stablehlo.multiply %314, %305 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %317 = stablehlo.add %315, %316 : tensor<1x14x14x1536xf32>
      %318 = stablehlo.multiply %317, %305 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %320 = stablehlo.add %318, %319 : tensor<1x14x14x1536xf32>
      %321 = stablehlo.multiply %320, %305 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %322 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %323 = stablehlo.add %321, %322 : tensor<1x14x14x1536xf32>
      %324 = stablehlo.multiply %300, %323 : tensor<1x14x14x1536xf32>
      %325 = stablehlo.subtract %302, %324 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %327 = stablehlo.compare LT, %300, %326 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %328 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %329 = stablehlo.negate %305 : tensor<1x14x14x1536xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %330 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %331 = stablehlo.compare LT, %329, %330 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %332 = stablehlo.exponential %329 : tensor<1x14x14x1536xf32>
      %333 = stablehlo.divide %302, %301 : tensor<1x14x14x1536xf32>
      %334 = stablehlo.multiply %332, %333 : tensor<1x14x14x1536xf32>
      %335 = stablehlo.compare LT, %301, %328 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %337 = stablehlo.divide %302, %305 : tensor<1x14x14x1536xf32>
      %338 = stablehlo.multiply %336, %337 : tensor<1x14x14x1536xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %340 = stablehlo.add %338, %339 : tensor<1x14x14x1536xf32>
      %341 = stablehlo.multiply %340, %337 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %343 = stablehlo.add %341, %342 : tensor<1x14x14x1536xf32>
      %344 = stablehlo.multiply %343, %337 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %345 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %346 = stablehlo.add %344, %345 : tensor<1x14x14x1536xf32>
      %347 = stablehlo.multiply %346, %337 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %349 = stablehlo.add %347, %348 : tensor<1x14x14x1536xf32>
      %350 = stablehlo.multiply %349, %337 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %351 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %352 = stablehlo.add %350, %351 : tensor<1x14x14x1536xf32>
      %353 = stablehlo.multiply %352, %337 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %354 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %355 = stablehlo.add %353, %354 : tensor<1x14x14x1536xf32>
      %356 = stablehlo.multiply %355, %337 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %357 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %358 = stablehlo.add %356, %357 : tensor<1x14x14x1536xf32>
      %359 = stablehlo.multiply %358, %337 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %360 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %361 = stablehlo.add %359, %360 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %362 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %363 = stablehlo.multiply %362, %337 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %364 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %365 = stablehlo.add %363, %364 : tensor<1x14x14x1536xf32>
      %366 = stablehlo.multiply %365, %337 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %367 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %368 = stablehlo.add %366, %367 : tensor<1x14x14x1536xf32>
      %369 = stablehlo.multiply %368, %337 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %371 = stablehlo.add %369, %370 : tensor<1x14x14x1536xf32>
      %372 = stablehlo.multiply %371, %337 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %373 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %374 = stablehlo.add %372, %373 : tensor<1x14x14x1536xf32>
      %375 = stablehlo.multiply %374, %337 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %376 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %377 = stablehlo.add %375, %376 : tensor<1x14x14x1536xf32>
      %378 = stablehlo.multiply %377, %337 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %379 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %380 = stablehlo.add %378, %379 : tensor<1x14x14x1536xf32>
      %381 = stablehlo.multiply %380, %337 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %382 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %383 = stablehlo.add %381, %382 : tensor<1x14x14x1536xf32>
      %384 = stablehlo.select %335, %361, %383 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %385 = stablehlo.multiply %334, %384 : tensor<1x14x14x1536xf32>
      %386 = stablehlo.select %331, %326, %385 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %387 = stablehlo.subtract %328, %386 : tensor<1x14x14x1536xf32>
      %388 = stablehlo.select %327, %387, %386 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %389 = stablehlo.select %303, %325, %388 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %296, %389 : tensor<1x14x14x1536xf32>
      mhlo.return %390 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %229 = "mhlo.fusion"(%228) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<196x1536xf32>, tensor<f32>) -> tensor<1536xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1536xf32>) -> tensor<1x1536xf32>
      %296 = stablehlo.sqrt %295 : tensor<1x1536xf32>
      mhlo.return %296 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x1536xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1536xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x1536xf32>) -> tensor<f32>
    %231 = "mhlo.fusion"(%228, %arg158, %arg159, %229, %230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1536xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x1536xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x1536xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1536xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x14x14x1536xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x14x14x1536xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x14x14x1536xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x14x14x1536xf32>
      mhlo.return %307 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1536xf32>, tensor<f32>) -> tensor<1x14x14x1536xf32>
    %232 = mhlo.bitcast %231 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    %233 = "mhlo.fusion"(%232, %arg160) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<384x1536xf32>):
      %292 = stablehlo.dot_general %arg201, %arg202, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
      mhlo.return %292 : tensor<196x384xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[196,384]{0,1}"} : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %234 = "mhlo.fusion"(%219, %233, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<384xf32>) -> tensor<384x196xf32>
      %293 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<384x196xf32>
      %294 = stablehlo.add %292, %293 : tensor<384x196xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<384x196xf32>) -> tensor<1x384x14x14xf32>
      %296 = stablehlo.add %arg201, %295 : tensor<1x384x14x14xf32>
      mhlo.return %296 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x384x14x14xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %293 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %236 = "mhlo.fusion"(%234, %235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>, %arg202: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x384x14x14xf32>
      mhlo.return %297 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %237 = "mhlo.fusion"(%236) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x384x14x14xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x384x14x14xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x384x14x14xf32>) -> tensor<384x14x14xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<384x14x14xf32>, tensor<f32>) -> tensor<14x14xf32>
      mhlo.return %294 : tensor<14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x384x14x14xf32>) -> tensor<14x14xf32>
    %238 = "mhlo.fusion"(%arg163, %arg162, %236, %237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x384x14x14xf32>, %arg204: tensor<14x14xf32>):
      %292 = mhlo.bitcast %arg204 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<14x14xf32>) -> tensor<1x14x14xf32>
      %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x14x14xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x14x14xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x14x14xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
      %300 = stablehlo.multiply %arg203, %299 : tensor<1x384x14x14xf32>
      %301 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<1x384x14x14xf32>
      %303 = stablehlo.broadcast_in_dim %arg201, dims = [1] : (tensor<384xf32>) -> tensor<1x384x14x14xf32>
      %304 = stablehlo.add %302, %303 : tensor<1x384x14x14xf32>
      mhlo.return %304 : tensor<1x384x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x384x14x14xf32>, tensor<14x14xf32>) -> tensor<1x384x14x14xf32>
    %239 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%238, %arg164, %arg165) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x384x14x14xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>) -> tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>
    %240 = stablehlo.get_tuple_element %239[0] : (tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>) -> tensor<1x768x7x7xf32>
    %241 = stablehlo.custom_call @__cudnn$convForward(%240, %arg166) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>
    %242 = stablehlo.get_tuple_element %241[0] : (tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>) -> tensor<1x768x7x7xf32>
    %243 = "mhlo.fusion"(%242, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1x768x7x7xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x768x7x7xf32>
      mhlo.return %293 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
      mhlo.return %293 : tensor<7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>) -> tensor<7x7xf32>
    %245 = "mhlo.fusion"(%243, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<7x7xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x7x7xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x768x7x7xf32>
      mhlo.return %297 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %246 = "mhlo.fusion"(%245) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x7x7xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
      mhlo.return %294 : tensor<7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>) -> tensor<7x7xf32>
    %247 = "mhlo.fusion"(%245, %246) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<7x7xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x7x7xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x7x7xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x7x7xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x768x7x7xf32>
      mhlo.return %300 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %248 = mhlo.bitcast %247 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<1x768x7x7xf32>) -> tensor<49x768xf32>
    %249 = "mhlo.fusion"(%248, %arg168, %arg169, %arg170) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<49x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
      %296 = stablehlo.dot_general %arg204, %295, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<3072x768xf32>, tensor<49x768xf32>) -> tensor<3072x49xf32>
      mhlo.return %296 : tensor<3072x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<49x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x49xf32>
    %250:2 = "mhlo.fusion"(%249, %arg171) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3072x49xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x49xf32>) -> tensor<49x3072xf32>
      %295 = stablehlo.add %293, %294 : tensor<49x3072xf32>
      %296 = stablehlo.multiply %292, %295 : tensor<49x3072xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %298 = stablehlo.negate %295 : tensor<49x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<49x3072xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %302 = stablehlo.abs %301 : tensor<1x7x7x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %304 = stablehlo.compare LT, %302, %303 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %306 = stablehlo.multiply %301, %301 : tensor<1x7x7x3072xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<1x7x7x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %309 = stablehlo.add %307, %308 : tensor<1x7x7x3072xf32>
      %310 = stablehlo.multiply %309, %306 : tensor<1x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %312 = stablehlo.add %310, %311 : tensor<1x7x7x3072xf32>
      %313 = stablehlo.multiply %312, %306 : tensor<1x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %315 = stablehlo.add %313, %314 : tensor<1x7x7x3072xf32>
      %316 = stablehlo.multiply %315, %306 : tensor<1x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %318 = stablehlo.add %316, %317 : tensor<1x7x7x3072xf32>
      %319 = stablehlo.multiply %318, %306 : tensor<1x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %321 = stablehlo.add %319, %320 : tensor<1x7x7x3072xf32>
      %322 = stablehlo.multiply %321, %306 : tensor<1x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %323 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %324 = stablehlo.add %322, %323 : tensor<1x7x7x3072xf32>
      %325 = stablehlo.multiply %301, %324 : tensor<1x7x7x3072xf32>
      %326 = stablehlo.subtract %303, %325 : tensor<1x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %328 = stablehlo.compare LT, %301, %327 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %329 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %330 = stablehlo.negate %306 : tensor<1x7x7x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %332 = stablehlo.compare LT, %330, %331 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %333 = stablehlo.exponential %330 : tensor<1x7x7x3072xf32>
      %334 = stablehlo.divide %303, %302 : tensor<1x7x7x3072xf32>
      %335 = stablehlo.multiply %333, %334 : tensor<1x7x7x3072xf32>
      %336 = stablehlo.compare LT, %302, %329 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %338 = stablehlo.divide %303, %306 : tensor<1x7x7x3072xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<1x7x7x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %341 = stablehlo.add %339, %340 : tensor<1x7x7x3072xf32>
      %342 = stablehlo.multiply %341, %338 : tensor<1x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %343 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %344 = stablehlo.add %342, %343 : tensor<1x7x7x3072xf32>
      %345 = stablehlo.multiply %344, %338 : tensor<1x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %347 = stablehlo.add %345, %346 : tensor<1x7x7x3072xf32>
      %348 = stablehlo.multiply %347, %338 : tensor<1x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %350 = stablehlo.add %348, %349 : tensor<1x7x7x3072xf32>
      %351 = stablehlo.multiply %350, %338 : tensor<1x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %352 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %353 = stablehlo.add %351, %352 : tensor<1x7x7x3072xf32>
      %354 = stablehlo.multiply %353, %338 : tensor<1x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<1x7x7x3072xf32>
      %357 = stablehlo.multiply %356, %338 : tensor<1x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<1x7x7x3072xf32>
      %360 = stablehlo.multiply %359, %338 : tensor<1x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<1x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %364 = stablehlo.multiply %363, %338 : tensor<1x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<1x7x7x3072xf32>
      %367 = stablehlo.multiply %366, %338 : tensor<1x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<1x7x7x3072xf32>
      %370 = stablehlo.multiply %369, %338 : tensor<1x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<1x7x7x3072xf32>
      %373 = stablehlo.multiply %372, %338 : tensor<1x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<1x7x7x3072xf32>
      %376 = stablehlo.multiply %375, %338 : tensor<1x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %378 = stablehlo.add %376, %377 : tensor<1x7x7x3072xf32>
      %379 = stablehlo.multiply %378, %338 : tensor<1x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x7x7x3072xf32>
      %382 = stablehlo.multiply %381, %338 : tensor<1x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<1x7x7x3072xf32>
      %385 = stablehlo.select %336, %362, %384 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %386 = stablehlo.multiply %335, %385 : tensor<1x7x7x3072xf32>
      %387 = stablehlo.select %332, %327, %386 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %388 = stablehlo.subtract %329, %387 : tensor<1x7x7x3072xf32>
      %389 = stablehlo.select %328, %388, %387 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %390 = stablehlo.select %304, %326, %389 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %391 = stablehlo.multiply %297, %390 : tensor<1x7x7x3072xf32>
      %392 = stablehlo.multiply %391, %391 : tensor<1x7x7x3072xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
      %394 = stablehlo.reduce(%393 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<49x3072xf32>, tensor<f32>) -> tensor<3072xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
      %396 = stablehlo.sqrt %395 : tensor<1x3072xf32>
      mhlo.return %396, %391 : tensor<1x3072xf32>, tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x49xf32>, tensor<3072xf32>) -> (tensor<1x3072xf32>, tensor<1x7x7x3072xf32>)
    %251 = "mhlo.fusion"(%250#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x3072xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3072xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x3072xf32>) -> tensor<f32>
    %252 = "mhlo.fusion"(%250#1, %arg172, %arg173, %250#0, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<1x1x1x3072xf32>, %arg204: tensor<1x3072xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x3072xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x7x7x3072xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x7x7x3072xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x7x7x3072xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x7x7x3072xf32>
      mhlo.return %307 : tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x3072xf32>, tensor<f32>) -> tensor<1x7x7x3072xf32>
    %253 = mhlo.bitcast %252 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    %254 = "mhlo.fusion"(%253, %arg174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<49x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<49x3072xf32>) -> tensor<768x49xf32>
      mhlo.return %292 : tensor<768x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x3072xf32>, tensor<768x3072xf32>) -> tensor<768x49xf32>
    %255 = "mhlo.fusion"(%240, %254, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768x49xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x49xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<768x49xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x49xf32>) -> tensor<1x768x7x7xf32>
      %295 = stablehlo.add %arg201, %294 : tensor<1x768x7x7xf32>
      mhlo.return %295 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768x49xf32>, tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %256 = stablehlo.custom_call @__cudnn$convForward(%255, %arg176) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>
    %257 = stablehlo.get_tuple_element %256[0] : (tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>) -> tensor<1x768x7x7xf32>
    %258 = "mhlo.fusion"(%257, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1x768x7x7xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x768x7x7xf32>
      mhlo.return %293 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
      mhlo.return %293 : tensor<7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>) -> tensor<7x7xf32>
    %260 = "mhlo.fusion"(%258, %259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<7x7xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x7x7xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x768x7x7xf32>
      mhlo.return %297 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %261 = "mhlo.fusion"(%260) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x7x7xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
      mhlo.return %294 : tensor<7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>) -> tensor<7x7xf32>
    %262 = "mhlo.fusion"(%260, %261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<7x7xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x7x7xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x7x7xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x7x7xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x768x7x7xf32>
      mhlo.return %300 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %263 = mhlo.bitcast %262 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<1x768x7x7xf32>) -> tensor<49x768xf32>
    %264 = "mhlo.fusion"(%263, %arg178, %arg179, %arg180) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<49x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
      %296 = stablehlo.dot_general %arg204, %295, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<3072x768xf32>, tensor<49x768xf32>) -> tensor<3072x49xf32>
      mhlo.return %296 : tensor<3072x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<49x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x49xf32>
    %265:2 = "mhlo.fusion"(%264, %arg181) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3072x49xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x49xf32>) -> tensor<49x3072xf32>
      %295 = stablehlo.add %293, %294 : tensor<49x3072xf32>
      %296 = stablehlo.multiply %292, %295 : tensor<49x3072xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %298 = stablehlo.negate %295 : tensor<49x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<49x3072xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %302 = stablehlo.abs %301 : tensor<1x7x7x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %304 = stablehlo.compare LT, %302, %303 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %306 = stablehlo.multiply %301, %301 : tensor<1x7x7x3072xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<1x7x7x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %309 = stablehlo.add %307, %308 : tensor<1x7x7x3072xf32>
      %310 = stablehlo.multiply %309, %306 : tensor<1x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %312 = stablehlo.add %310, %311 : tensor<1x7x7x3072xf32>
      %313 = stablehlo.multiply %312, %306 : tensor<1x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %315 = stablehlo.add %313, %314 : tensor<1x7x7x3072xf32>
      %316 = stablehlo.multiply %315, %306 : tensor<1x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %318 = stablehlo.add %316, %317 : tensor<1x7x7x3072xf32>
      %319 = stablehlo.multiply %318, %306 : tensor<1x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %321 = stablehlo.add %319, %320 : tensor<1x7x7x3072xf32>
      %322 = stablehlo.multiply %321, %306 : tensor<1x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %323 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %324 = stablehlo.add %322, %323 : tensor<1x7x7x3072xf32>
      %325 = stablehlo.multiply %301, %324 : tensor<1x7x7x3072xf32>
      %326 = stablehlo.subtract %303, %325 : tensor<1x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %328 = stablehlo.compare LT, %301, %327 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %329 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %330 = stablehlo.negate %306 : tensor<1x7x7x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %332 = stablehlo.compare LT, %330, %331 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %333 = stablehlo.exponential %330 : tensor<1x7x7x3072xf32>
      %334 = stablehlo.divide %303, %302 : tensor<1x7x7x3072xf32>
      %335 = stablehlo.multiply %333, %334 : tensor<1x7x7x3072xf32>
      %336 = stablehlo.compare LT, %302, %329 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %338 = stablehlo.divide %303, %306 : tensor<1x7x7x3072xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<1x7x7x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %341 = stablehlo.add %339, %340 : tensor<1x7x7x3072xf32>
      %342 = stablehlo.multiply %341, %338 : tensor<1x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %343 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %344 = stablehlo.add %342, %343 : tensor<1x7x7x3072xf32>
      %345 = stablehlo.multiply %344, %338 : tensor<1x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %347 = stablehlo.add %345, %346 : tensor<1x7x7x3072xf32>
      %348 = stablehlo.multiply %347, %338 : tensor<1x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %350 = stablehlo.add %348, %349 : tensor<1x7x7x3072xf32>
      %351 = stablehlo.multiply %350, %338 : tensor<1x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %352 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %353 = stablehlo.add %351, %352 : tensor<1x7x7x3072xf32>
      %354 = stablehlo.multiply %353, %338 : tensor<1x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<1x7x7x3072xf32>
      %357 = stablehlo.multiply %356, %338 : tensor<1x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<1x7x7x3072xf32>
      %360 = stablehlo.multiply %359, %338 : tensor<1x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<1x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %364 = stablehlo.multiply %363, %338 : tensor<1x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<1x7x7x3072xf32>
      %367 = stablehlo.multiply %366, %338 : tensor<1x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<1x7x7x3072xf32>
      %370 = stablehlo.multiply %369, %338 : tensor<1x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<1x7x7x3072xf32>
      %373 = stablehlo.multiply %372, %338 : tensor<1x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<1x7x7x3072xf32>
      %376 = stablehlo.multiply %375, %338 : tensor<1x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %378 = stablehlo.add %376, %377 : tensor<1x7x7x3072xf32>
      %379 = stablehlo.multiply %378, %338 : tensor<1x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x7x7x3072xf32>
      %382 = stablehlo.multiply %381, %338 : tensor<1x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<1x7x7x3072xf32>
      %385 = stablehlo.select %336, %362, %384 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %386 = stablehlo.multiply %335, %385 : tensor<1x7x7x3072xf32>
      %387 = stablehlo.select %332, %327, %386 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %388 = stablehlo.subtract %329, %387 : tensor<1x7x7x3072xf32>
      %389 = stablehlo.select %328, %388, %387 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %390 = stablehlo.select %304, %326, %389 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %391 = stablehlo.multiply %297, %390 : tensor<1x7x7x3072xf32>
      %392 = stablehlo.multiply %391, %391 : tensor<1x7x7x3072xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
      %394 = stablehlo.reduce(%393 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<49x3072xf32>, tensor<f32>) -> tensor<3072xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
      %396 = stablehlo.sqrt %395 : tensor<1x3072xf32>
      mhlo.return %396, %391 : tensor<1x3072xf32>, tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x49xf32>, tensor<3072xf32>) -> (tensor<1x3072xf32>, tensor<1x7x7x3072xf32>)
    %266 = "mhlo.fusion"(%265#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x3072xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3072xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x3072xf32>) -> tensor<f32>
    %267 = "mhlo.fusion"(%265#1, %arg182, %arg183, %265#0, %266) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<1x1x1x3072xf32>, %arg204: tensor<1x3072xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x3072xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x7x7x3072xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x7x7x3072xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x7x7x3072xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x7x7x3072xf32>
      mhlo.return %307 : tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x3072xf32>, tensor<f32>) -> tensor<1x7x7x3072xf32>
    %268 = mhlo.bitcast %267 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    %269 = "mhlo.fusion"(%268, %arg184) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<49x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<49x3072xf32>) -> tensor<768x49xf32>
      mhlo.return %292 : tensor<768x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x3072xf32>, tensor<768x3072xf32>) -> tensor<768x49xf32>
    %270 = "mhlo.fusion"(%255, %269, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768x49xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x49xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<768x49xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x49xf32>) -> tensor<1x768x7x7xf32>
      %295 = stablehlo.add %arg201, %294 : tensor<1x768x7x7xf32>
      mhlo.return %295 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768x49xf32>, tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %271 = stablehlo.custom_call @__cudnn$convForward(%270, %arg186) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1,768,7,7]{3,2,1,0}, u8[0]{0})"} : (tensor<1x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>
    %272 = stablehlo.get_tuple_element %271[0] : (tuple<tensor<1x768x7x7xf32>, tensor<0xui8>>) -> tensor<1x768x7x7xf32>
    %273 = "mhlo.fusion"(%272, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<1x768x7x7xf32>
      %293 = stablehlo.add %arg201, %292 : tensor<1x768x7x7xf32>
      mhlo.return %293 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768xf32>) -> tensor<1x768x7x7xf32>
    %274 = "mhlo.fusion"(%273) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
      mhlo.return %293 : tensor<7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>) -> tensor<7x7xf32>
    %275 = "mhlo.fusion"(%273, %274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<7x7xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x7x7xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %296 = stablehlo.broadcast_in_dim %295, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
      %297 = stablehlo.subtract %arg201, %296 : tensor<1x768x7x7xf32>
      mhlo.return %297 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %276 = "mhlo.fusion"(%275) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>):
      %292 = stablehlo.multiply %arg201, %arg201 : tensor<1x768x7x7xf32>
      %293 = mhlo.bitcast %292 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x7x7xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %294 = stablehlo.reduce(%293 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<768x7x7xf32>, tensor<f32>) -> tensor<7x7xf32>
      mhlo.return %294 : tensor<7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>) -> tensor<7x7xf32>
    %277 = "mhlo.fusion"(%275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<7x7xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<7x7xf32>) -> tensor<1x7x7xf32>
      %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x7x7xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %295 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %296 = stablehlo.add %294, %295 : tensor<1x7x7xf32>
      %297 = stablehlo.rsqrt %296 : tensor<1x7x7xf32>
      %298 = mhlo.bitcast %297 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %299 = stablehlo.broadcast_in_dim %298, dims = [2, 3] : (tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
      %300 = stablehlo.multiply %arg201, %299 : tensor<1x768x7x7xf32>
      mhlo.return %300 : tensor<1x768x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<7x7xf32>) -> tensor<1x768x7x7xf32>
    %278 = mhlo.bitcast %277 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<1x768x7x7xf32>) -> tensor<49x768xf32>
    %279 = "mhlo.fusion"(%278, %arg188, %arg189, %arg190) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<49x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<3072x768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg202, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
      %293 = stablehlo.multiply %arg201, %292 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
      %294 = stablehlo.broadcast_in_dim %arg203, dims = [1] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : (tensor<768xf32>) -> tensor<49x768xf32>
      %295 = stablehlo.add %293, %294 {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[49,768]{0,1}"} : tensor<49x768xf32>
      %296 = stablehlo.dot_general %arg204, %295, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<3072x768xf32>, tensor<49x768xf32>) -> tensor<3072x49xf32>
      mhlo.return %296 : tensor<3072x49xf32>
    }) {output_operand_aliasing = [], result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[3072,49]{0,1}"} : (tensor<49x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768xf32>) -> tensor<3072x49xf32>
    %280:2 = "mhlo.fusion"(%279, %arg191) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<3072x49xf32>, %arg202: tensor<3072xf32>):
      %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %292 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %293 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
      %294 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<3072x49xf32>) -> tensor<49x3072xf32>
      %295 = stablehlo.add %293, %294 : tensor<49x3072xf32>
      %296 = stablehlo.multiply %292, %295 : tensor<49x3072xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %298 = stablehlo.negate %295 : tensor<49x3072xf32>
      %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %300 = stablehlo.multiply %298, %299 : tensor<49x3072xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %302 = stablehlo.abs %301 : tensor<1x7x7x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %304 = stablehlo.compare LT, %302, %303 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %306 = stablehlo.multiply %301, %301 : tensor<1x7x7x3072xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<1x7x7x3072xf32>
      %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %309 = stablehlo.add %307, %308 : tensor<1x7x7x3072xf32>
      %310 = stablehlo.multiply %309, %306 : tensor<1x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %311 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %312 = stablehlo.add %310, %311 : tensor<1x7x7x3072xf32>
      %313 = stablehlo.multiply %312, %306 : tensor<1x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %315 = stablehlo.add %313, %314 : tensor<1x7x7x3072xf32>
      %316 = stablehlo.multiply %315, %306 : tensor<1x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %318 = stablehlo.add %316, %317 : tensor<1x7x7x3072xf32>
      %319 = stablehlo.multiply %318, %306 : tensor<1x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %321 = stablehlo.add %319, %320 : tensor<1x7x7x3072xf32>
      %322 = stablehlo.multiply %321, %306 : tensor<1x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %323 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %324 = stablehlo.add %322, %323 : tensor<1x7x7x3072xf32>
      %325 = stablehlo.multiply %301, %324 : tensor<1x7x7x3072xf32>
      %326 = stablehlo.subtract %303, %325 : tensor<1x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %327 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %328 = stablehlo.compare LT, %301, %327 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %329 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %330 = stablehlo.negate %306 : tensor<1x7x7x3072xf32>
      %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %331 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %332 = stablehlo.compare LT, %330, %331 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %333 = stablehlo.exponential %330 : tensor<1x7x7x3072xf32>
      %334 = stablehlo.divide %303, %302 : tensor<1x7x7x3072xf32>
      %335 = stablehlo.multiply %333, %334 : tensor<1x7x7x3072xf32>
      %336 = stablehlo.compare LT, %302, %329 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %338 = stablehlo.divide %303, %306 : tensor<1x7x7x3072xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<1x7x7x3072xf32>
      %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %341 = stablehlo.add %339, %340 : tensor<1x7x7x3072xf32>
      %342 = stablehlo.multiply %341, %338 : tensor<1x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %343 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %344 = stablehlo.add %342, %343 : tensor<1x7x7x3072xf32>
      %345 = stablehlo.multiply %344, %338 : tensor<1x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %346 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %347 = stablehlo.add %345, %346 : tensor<1x7x7x3072xf32>
      %348 = stablehlo.multiply %347, %338 : tensor<1x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %350 = stablehlo.add %348, %349 : tensor<1x7x7x3072xf32>
      %351 = stablehlo.multiply %350, %338 : tensor<1x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %352 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %353 = stablehlo.add %351, %352 : tensor<1x7x7x3072xf32>
      %354 = stablehlo.multiply %353, %338 : tensor<1x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %356 = stablehlo.add %354, %355 : tensor<1x7x7x3072xf32>
      %357 = stablehlo.multiply %356, %338 : tensor<1x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %358 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %359 = stablehlo.add %357, %358 : tensor<1x7x7x3072xf32>
      %360 = stablehlo.multiply %359, %338 : tensor<1x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %361 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %362 = stablehlo.add %360, %361 : tensor<1x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %363 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %364 = stablehlo.multiply %363, %338 : tensor<1x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %365 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %366 = stablehlo.add %364, %365 : tensor<1x7x7x3072xf32>
      %367 = stablehlo.multiply %366, %338 : tensor<1x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %368 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %369 = stablehlo.add %367, %368 : tensor<1x7x7x3072xf32>
      %370 = stablehlo.multiply %369, %338 : tensor<1x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %371 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %372 = stablehlo.add %370, %371 : tensor<1x7x7x3072xf32>
      %373 = stablehlo.multiply %372, %338 : tensor<1x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %374 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %375 = stablehlo.add %373, %374 : tensor<1x7x7x3072xf32>
      %376 = stablehlo.multiply %375, %338 : tensor<1x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %377 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %378 = stablehlo.add %376, %377 : tensor<1x7x7x3072xf32>
      %379 = stablehlo.multiply %378, %338 : tensor<1x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %380 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %381 = stablehlo.add %379, %380 : tensor<1x7x7x3072xf32>
      %382 = stablehlo.multiply %381, %338 : tensor<1x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %383 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %384 = stablehlo.add %382, %383 : tensor<1x7x7x3072xf32>
      %385 = stablehlo.select %336, %362, %384 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %386 = stablehlo.multiply %335, %385 : tensor<1x7x7x3072xf32>
      %387 = stablehlo.select %332, %327, %386 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %388 = stablehlo.subtract %329, %387 : tensor<1x7x7x3072xf32>
      %389 = stablehlo.select %328, %388, %387 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %390 = stablehlo.select %304, %326, %389 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %391 = stablehlo.multiply %297, %390 : tensor<1x7x7x3072xf32>
      %392 = stablehlo.multiply %391, %391 : tensor<1x7x7x3072xf32>
      %393 = mhlo.bitcast %392 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
      %394 = stablehlo.reduce(%393 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<49x3072xf32>, tensor<f32>) -> tensor<3072xf32>
      %395 = mhlo.bitcast %394 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<3072xf32>) -> tensor<1x3072xf32>
      %396 = stablehlo.sqrt %395 : tensor<1x3072xf32>
      mhlo.return %396, %391 : tensor<1x3072xf32>, tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<3072x49xf32>, tensor<3072xf32>) -> (tensor<1x3072xf32>, tensor<1x7x7x3072xf32>)
    %281 = "mhlo.fusion"(%280#0) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x3072xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %293 = stablehlo.reduce(%292 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3072xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %293 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<1x3072xf32>) -> tensor<f32>
    %282 = "mhlo.fusion"(%280#1, %arg192, %arg193, %280#0, %281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<1x1x1x3072xf32>, %arg204: tensor<1x3072xf32>, %arg205: tensor<f32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %294 = mhlo.bitcast %arg205 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1xf32>
      %295 = stablehlo.multiply %294, %cst : tensor<1xf32>
      %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1xf32>
      %296 = stablehlo.add %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x3072xf32>
      %299 = stablehlo.divide %arg204, %298 : tensor<1x3072xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<3072xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %302 = stablehlo.multiply %arg201, %301 : tensor<1x7x7x3072xf32>
      %303 = stablehlo.multiply %293, %302 : tensor<1x7x7x3072xf32>
      %304 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %306 = stablehlo.add %303, %305 : tensor<1x7x7x3072xf32>
      %307 = stablehlo.add %306, %arg201 : tensor<1x7x7x3072xf32>
      mhlo.return %307 : tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x3072xf32>, tensor<f32>) -> tensor<1x7x7x3072xf32>
    %283 = mhlo.bitcast %282 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    %284 = "mhlo.fusion"(%283, %arg194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<49x3072xf32>, %arg202: tensor<768x3072xf32>):
      %292 = stablehlo.dot_general %arg202, %arg201, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<768x3072xf32>, tensor<49x3072xf32>) -> tensor<768x49xf32>
      mhlo.return %292 : tensor<768x49xf32>
    }) {output_operand_aliasing = []} : (tensor<49x3072xf32>, tensor<768x3072xf32>) -> tensor<768x49xf32>
    %285 = "mhlo.fusion"(%270, %284, %arg195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1x768x7x7xf32>, %arg202: tensor<768x49xf32>, %arg203: tensor<768xf32>):
      %292 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<768xf32>) -> tensor<768x49xf32>
      %293 = stablehlo.add %292, %arg202 : tensor<768x49xf32>
      %294 = mhlo.bitcast %293 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<768x49xf32>) -> tensor<1x768x7x7xf32>
      %295 = stablehlo.add %arg201, %294 : tensor<1x768x7x7xf32>
      %296 = mhlo.bitcast %295 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x7x7xf32>) -> tensor<768x49xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %297 = stablehlo.reduce(%296 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<768x49xf32>, tensor<f32>) -> tensor<768xf32>
      mhlo.return %297 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768x7x7xf32>, tensor<768x49xf32>, tensor<768xf32>) -> tensor<768xf32>
    %286 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<768xf32>):
      %292 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x768xf32>
      %295 = mhlo.bitcast %294 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %296 = stablehlo.reduce(%295 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %296 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<f32>
    %287 = "mhlo.fusion"(%286, %285) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<f32>, %arg202: tensor<768xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x768xf32>
      %295 = mhlo.bitcast %arg201 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %296 = stablehlo.multiply %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %299 = stablehlo.subtract %294, %298 : tensor<1x768xf32>
      %300 = stablehlo.multiply %299, %299 : tensor<1x768xf32>
      %301 = mhlo.bitcast %300 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %302 = stablehlo.reduce(%301 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<768xf32>, tensor<f32>) -> tensor<f32>
      mhlo.return %302 : tensor<f32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<768xf32>) -> tensor<f32>
    %288 = "mhlo.fusion"(%arg197, %arg196, %287, %286, %285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<f32>, %arg204: tensor<f32>, %arg205: tensor<768xf32>):
      %292 = mhlo.bitcast %arg205 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %293 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %294 = stablehlo.multiply %292, %293 : tensor<1x768xf32>
      %295 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %296 = stablehlo.multiply %295, %cst_0 : tensor<1xf32>
      %297 = mhlo.bitcast %296 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %299 = stablehlo.subtract %294, %298 : tensor<1x768xf32>
      %300 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<> : tensor<0xindex>} : (tensor<f32>) -> tensor<1xf32>
      %301 = stablehlo.multiply %300, %cst_0 : tensor<1xf32>
      %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<1xf32>
      %302 = stablehlo.add %301, %cst_1 : tensor<1xf32>
      %303 = stablehlo.rsqrt %302 : tensor<1xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %306 = stablehlo.multiply %299, %305 : tensor<1x768xf32>
      %307 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<1x768xf32>
      %309 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %310 = stablehlo.add %308, %309 : tensor<1x768xf32>
      mhlo.return %310 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<f32>, tensor<f32>, tensor<768xf32>) -> tensor<1x768xf32>
    %289 = "mhlo.fusion"(%arg198, %288) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg201: tensor<1000x768xf32>, %arg202: tensor<1x768xf32>):
      %292 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      %293 = stablehlo.broadcast_in_dim %292, dims = [1] : (tensor<768xf32>) -> tensor<1000x768xf32>
      %294 = stablehlo.multiply %293, %arg201 : tensor<1000x768xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %295 = stablehlo.reduce(%294 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1000x768xf32>, tensor<f32>) -> tensor<1000xf32>
      mhlo.return %295 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x768xf32>, tensor<1x768xf32>) -> tensor<1000xf32>
    %290 = "mhlo.fusion"(%arg199, %289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1000xf32>, %arg202: tensor<1000xf32>):
      %292 = stablehlo.add %arg201, %arg202 : tensor<1000xf32>
      mhlo.return %292 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<1000xf32>) -> tensor<1000xf32>
    %291 = mhlo.bitcast %290 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %291 : tensor<1x1000xf32>
  }
}
