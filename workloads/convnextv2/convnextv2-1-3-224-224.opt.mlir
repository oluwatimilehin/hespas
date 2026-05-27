module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_66.75(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x768xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x768xf32>
    return %0 : tensor<1x768xf32>
  }
  func.func private @region_63.72(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    return %0 : tensor<1x7x7xf32>
  }
  func.func private @region_60.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    return %0 : tensor<1x7x7xf32>
  }
  func.func private @region_55.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    return %0 : tensor<1x7x7xf32>
  }
  func.func private @region_54.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_51.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_48.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_45.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_42.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_39.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_36.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_33.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_30.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_25.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    return %0 : tensor<1x14x14xf32>
  }
  func.func private @region_24.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    return %0 : tensor<1x28x28xf32>
  }
  func.func private @region_21.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    return %0 : tensor<1x28x28xf32>
  }
  func.func private @region_18.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    return %0 : tensor<1x28x28xf32>
  }
  func.func private @region_13.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    return %0 : tensor<1x28x28xf32>
  }
  func.func private @region_12.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    return %0 : tensor<1x56x56xf32>
  }
  func.func private @region_9.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    return %0 : tensor<1x56x56xf32>
  }
  func.func private @region_6.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    return %0 : tensor<1x56x56xf32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    return %0 : tensor<1x56x56xf32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    return %0 : tensor<1x56x56xf32>
  }
  func.func private @fused_computation.23(%arg0: tensor<1000xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768x1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.dot %arg1, %arg2, precision = [DEFAULT, DEFAULT] : (tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func.func private @fused_computation.24(%arg0: tensor<1000x768xf32>) -> tensor<768x1000xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,1000]{0,1}"} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
    %1 = mhlo.copy %0 : tensor<768x1000xf32>
    return %1 : tensor<768x1000xf32>
  }
  func.func private @fused_computation.25(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1x768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x768xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %2 = stablehlo.multiply %arg3, %cst_0 : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<1x768xf32>
    %6 = mhlo.bitcast %arg2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %8 = stablehlo.multiply %5, %7 : tensor<1x768xf32>
    %9 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x768xf32>
    %11 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
    %12 = stablehlo.add %10, %11 : tensor<1x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
    return %13 : tensor<768xf32>
  }
  func.func private @region_69.78.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<1x24xf32>) -> tensor<1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x24xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %1 = stablehlo.multiply %0, %cst_0 : tensor<1xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<1xf32>
    %2 = stablehlo.add %1, %cst_1 : tensor<1xf32>
    %3 = stablehlo.rsqrt %2 : tensor<1xf32>
    return %3 : tensor<1xf32>
  }
  func.func private @fused_computation.27(%arg0: tensor<1xf32>, %arg1: tensor<1x768xf32>) -> tensor<1x768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x768xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
    %2 = stablehlo.multiply %arg0, %cst_0 : tensor<1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<1x768xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x768xf32>
    return %6 : tensor<1x768xf32>
  }
  func.func private @fused_computation.28(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1x768xf32>
    return %1 : tensor<1x768xf32>
  }
  func.func private @fused_computation.29(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<49x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<49x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<49x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x768xf32>) -> tensor<1x7x7x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x7x7x768xf32>
    return %3 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.30(%arg0: tensor<1x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<1x1x1x3072xf32>, %arg3: tensor<1x1x1x3072xf32>) -> tensor<49x3072xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x7x7x3072xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x7x7x3072xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x7x7x3072xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x7x7x3072xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    return %10 : tensor<49x3072xf32>
  }
  func.func private @fused_computation.31(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x3072xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x3072xf32>
    return %5 : tensor<1x1x1x3072xf32>
  }
  func.func private @region_64.73(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<1x7x7x3072xf32>) -> tensor<1x3072xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x7x7x3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x3072xf32>, tensor<f32>) -> tensor<1x3072xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x3072xf32>
    return %2 : tensor<1x3072xf32>
  }
  func.func private @fused_computation.33(%arg0: tensor<49x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x7x7x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<49x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<49x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %5 = stablehlo.negate %2 : tensor<49x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<49x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x7x7x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x7x7x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x7x7x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x7x7x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x7x7x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x7x7x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x7x7x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x7x7x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x7x7x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x7x7x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x7x7x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x7x7x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x7x7x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x7x7x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x7x7x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x7x7x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x7x7x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x7x7x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x7x7x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x7x7x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x7x7x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x7x7x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x7x7x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x7x7x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x7x7x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x7x7x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x7x7x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x7x7x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x7x7x3072xf32>
    return %98 : tensor<1x7x7x3072xf32>
  }
  func.func private @fused_computation.34(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x7x7xf32>, %arg3: tensor<1x7x7xf32>, %arg4: tensor<1x7x7x768xf32>, %arg5: tensor<768xf32>) -> tensor<49x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x7x7x768xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x7x7x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x7x7x768xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x7x7x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x768xf32>) -> tensor<49x768xf32>
    return %14 : tensor<49x768xf32>
  }
  func.func private @region_57.64.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<1x7x7x24xf32>) -> tensor<1x7x7xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x24xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x7x7xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x7x7xf32>
    return %5 : tensor<1x7x7xf32>
  }
  func.func private @fused_computation.36(%arg0: tensor<1x7x7xf32>, %arg1: tensor<1x7x7x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x7x7x768xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x7x7x768xf32>
    return %7 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.37(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x7x7x768xf32>
    return %1 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.38(%arg0: tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x768xf32>
    return %1 : tensor<7x7x1x768xf32>
  }
  func.func private @fused_computation.39(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<49x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<49x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<49x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x768xf32>) -> tensor<1x7x7x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x7x7x768xf32>
    return %3 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.40(%arg0: tensor<1x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<1x1x1x3072xf32>, %arg3: tensor<1x1x1x3072xf32>) -> tensor<49x3072xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x7x7x3072xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x7x7x3072xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x7x7x3072xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x7x7x3072xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    return %10 : tensor<49x3072xf32>
  }
  func.func private @fused_computation.41(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x3072xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x3072xf32>
    return %5 : tensor<1x1x1x3072xf32>
  }
  func.func private @region_61.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.42(%arg0: tensor<1x7x7x3072xf32>) -> tensor<1x3072xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x7x7x3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x3072xf32>, tensor<f32>) -> tensor<1x3072xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x3072xf32>
    return %2 : tensor<1x3072xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<49x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x7x7x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<49x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<49x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %5 = stablehlo.negate %2 : tensor<49x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<49x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x7x7x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x7x7x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x7x7x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x7x7x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x7x7x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x7x7x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x7x7x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x7x7x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x7x7x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x7x7x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x7x7x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x7x7x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x7x7x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x7x7x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x7x7x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x7x7x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x7x7x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x7x7x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x7x7x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x7x7x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x7x7x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x7x7x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x7x7x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x7x7x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x7x7x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x7x7x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x7x7x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x7x7x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x7x7x3072xf32>
    return %98 : tensor<1x7x7x3072xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x7x7xf32>, %arg3: tensor<1x7x7xf32>, %arg4: tensor<1x7x7x768xf32>, %arg5: tensor<768xf32>) -> tensor<49x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x7x7x768xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x7x7x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x7x7x768xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x7x7x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x768xf32>) -> tensor<49x768xf32>
    return %14 : tensor<49x768xf32>
  }
  func.func private @region_57.64.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<1x7x7x24xf32>) -> tensor<1x7x7xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x24xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x7x7xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x7x7xf32>
    return %5 : tensor<1x7x7xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<1x7x7xf32>, %arg1: tensor<1x7x7x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x7x7x768xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x7x7x768xf32>
    return %7 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x7x7x768xf32>
    return %1 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.48(%arg0: tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x768xf32>
    return %1 : tensor<7x7x1x768xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<49x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x7x7x768xf32>, %arg3: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<1x7x7x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<49x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<49x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x768xf32>) -> tensor<1x7x7x768xf32>
    %5 = stablehlo.add %1, %4 : tensor<1x7x7x768xf32>
    return %5 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<1x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<1x1x1x3072xf32>, %arg3: tensor<1x1x1x3072xf32>) -> tensor<49x3072xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x7x7x3072xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x7x7x3072xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x7x7x3072xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x7x7x3072xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
    return %10 : tensor<49x3072xf32>
  }
  func.func private @fused_computation.51(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x3072xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x3072xf32>
    return %5 : tensor<1x1x1x3072xf32>
  }
  func.func private @region_58.67(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.52(%arg0: tensor<1x7x7x3072xf32>) -> tensor<1x3072xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x7x7x3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x3072xf32>, tensor<f32>) -> tensor<1x3072xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x3072xf32>
    return %2 : tensor<1x3072xf32>
  }
  func.func private @fused_computation.53(%arg0: tensor<49x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<1x7x7x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<49x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<49x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %5 = stablehlo.negate %2 : tensor<49x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<49x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
    %9 = stablehlo.abs %8 : tensor<1x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<1x7x7x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x7x7x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<1x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<1x7x7x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<1x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x7x7x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<1x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<1x7x7x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<1x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<1x7x7x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<1x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<1x7x7x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<1x7x7x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<1x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %37 = stablehlo.negate %13 : tensor<1x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<1x7x7x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<1x7x7x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<1x7x7x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<1x7x7x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<1x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<1x7x7x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<1x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<1x7x7x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<1x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x7x7x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<1x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<1x7x7x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<1x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x7x7x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<1x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x7x7x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<1x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<1x7x7x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<1x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<1x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<1x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<1x7x7x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<1x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<1x7x7x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<1x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<1x7x7x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<1x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<1x7x7x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<1x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<1x7x7x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<1x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<1x7x7x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<1x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<1x7x7x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<1x7x7x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<1x7x7x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<1x7x7x3072xf32>
    return %98 : tensor<1x7x7x3072xf32>
  }
  func.func private @fused_computation.54(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x7x7xf32>, %arg3: tensor<1x7x7xf32>, %arg4: tensor<1x7x7x768xf32>, %arg5: tensor<768xf32>) -> tensor<49x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x7x7x768xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x7x7x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x7x7x768xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x7x7x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x768xf32>) -> tensor<49x768xf32>
    return %14 : tensor<49x768xf32>
  }
  func.func private @region_57.64.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.55(%arg0: tensor<1x7x7x24xf32>) -> tensor<1x7x7xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x24xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x7x7xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x7x7xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x7x7xf32>
    return %5 : tensor<1x7x7xf32>
  }
  func.func private @fused_computation.56(%arg0: tensor<1x7x7xf32>, %arg1: tensor<1x7x7x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x7x7xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x7x7x768xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x7x7x768xf32>
    return %7 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x7x7x768xf32>
    return %1 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.58(%arg0: tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x768xf32>
    return %1 : tensor<7x7x1x768xf32>
  }
  func.func private @fused_computation.59(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x7x7x768xf32>
    return %1 : tensor<1x7x7x768xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,384,768]{1,0,2,3}"} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
    %1 = mhlo.copy %0 : tensor<2x2x384x768xf32>
    return %1 : tensor<2x2x384x768xf32>
  }
  func.func private @fused_computation.61(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14x384xf32>, %arg4: tensor<1x14x14xf32>) -> tensor<1x14x14x384xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x14x14x384xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x14x14x384xf32>
    return %11 : tensor<1x14x14x384xf32>
  }
  func.func private @region_27.32.clone.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.62(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<1x14x14xf32>) -> tensor<1x14x14x384xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x14x14xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x14x14x384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x14x14x384xf32>
    return %5 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_52.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.67(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.69(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.70(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_49.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.77(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.79(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.80(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_46.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.87(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.89(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.90(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_43.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.97(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.99(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.100(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_40.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.107(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.109(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.110(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_37.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.117(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.119(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.120(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_34.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.127(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.129(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.130(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<196x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<196x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x14x14x384xf32>
    return %3 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_31.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.137(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.139(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.140(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<196x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14x384xf32>, %arg3: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<1x14x14x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<196x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
    %5 = stablehlo.add %1, %4 : tensor<1x14x14x384xf32>
    return %5 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<1x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<1x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
    return %10 : tensor<196x1536xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x1536xf32>
    return %5 : tensor<1x1x1x1536xf32>
  }
  func.func private @region_28.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.147(%arg0: tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x1536xf32>
    return %2 : tensor<1x1536xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<196x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<1x14x14x1536xf32> {
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
  func.func private @fused_computation.149(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<1x14x14xf32>, %arg3: tensor<1x14x14xf32>, %arg4: tensor<1x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<196x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x14x14x384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x14x14x384xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
    return %14 : tensor<196x384xf32>
  }
  func.func private @region_27.32.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.150(%arg0: tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x14x14xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x14x14xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x14x14xf32>
    return %5 : tensor<1x14x14xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<1x14x14xf32>, %arg1: tensor<1x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x14x14x384xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x14x14x384xf32>
    return %7 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x14x14x384xf32>
    return %1 : tensor<1x14x14x384xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,192,384]{1,0,2,3}"} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
    %1 = mhlo.copy %0 : tensor<2x2x192x384xf32>
    return %1 : tensor<2x2x192x384xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x28x28xf32>, %arg3: tensor<1x28x28x192xf32>, %arg4: tensor<1x28x28xf32>) -> tensor<1x28x28x192xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x28x28xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x28x28x192xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x28x28x192xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x28x28x192xf32>
    return %11 : tensor<1x28x28x192xf32>
  }
  func.func private @region_15.18.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.157(%arg0: tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    return %5 : tensor<1x28x28xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<1x28x28xf32>) -> tensor<1x28x28x192xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x28x28xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x28x28x192xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x28x28x192xf32>
    return %5 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<784x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<784x192xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<784x192xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<1x28x28x192xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x28x28x192xf32>
    return %3 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<1x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<1x1x1x768xf32>, %arg3: tensor<1x1x1x768xf32>) -> tensor<784x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x28x28x768xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x28x28x768xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x28x28x768xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x28x28x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    return %10 : tensor<784x768xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x768xf32>) -> tensor<1x1x1x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x768xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x768xf32>
    return %5 : tensor<1x1x1x768xf32>
  }
  func.func private @region_22.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.162(%arg0: tensor<1x28x28x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x28x28x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x28x28x768xf32>, tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x768xf32>
    return %2 : tensor<1x768xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<784x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x28x28x768xf32> {
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
  func.func private @fused_computation.164(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x28x28xf32>, %arg3: tensor<1x28x28xf32>, %arg4: tensor<1x28x28x192xf32>, %arg5: tensor<192xf32>) -> tensor<784x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x28x28xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x28x28x192xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x28x28x192xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x28x28x192xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x192xf32>) -> tensor<784x192xf32>
    return %14 : tensor<784x192xf32>
  }
  func.func private @region_15.18.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.165(%arg0: tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    return %5 : tensor<1x28x28xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<1x28x28xf32>, %arg1: tensor<1x28x28x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x28x28xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x28x28x192xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x28x28x192xf32>
    return %7 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x28x28x192xf32>
    return %1 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x192xf32>
    return %1 : tensor<7x7x1x192xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<784x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<784x192xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<784x192xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<1x28x28x192xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x28x28x192xf32>
    return %3 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<1x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<1x1x1x768xf32>, %arg3: tensor<1x1x1x768xf32>) -> tensor<784x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x28x28x768xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x28x28x768xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x28x28x768xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x28x28x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    return %10 : tensor<784x768xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x768xf32>) -> tensor<1x1x1x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x768xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x768xf32>
    return %5 : tensor<1x1x1x768xf32>
  }
  func.func private @region_19.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.172(%arg0: tensor<1x28x28x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x28x28x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x28x28x768xf32>, tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x768xf32>
    return %2 : tensor<1x768xf32>
  }
  func.func private @fused_computation.173(%arg0: tensor<784x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x28x28x768xf32> {
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
  func.func private @fused_computation.174(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x28x28xf32>, %arg3: tensor<1x28x28xf32>, %arg4: tensor<1x28x28x192xf32>, %arg5: tensor<192xf32>) -> tensor<784x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x28x28xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x28x28x192xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x28x28x192xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x28x28x192xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x192xf32>) -> tensor<784x192xf32>
    return %14 : tensor<784x192xf32>
  }
  func.func private @region_15.18.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.175(%arg0: tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    return %5 : tensor<1x28x28xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<1x28x28xf32>, %arg1: tensor<1x28x28x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x28x28xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x28x28x192xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x28x28x192xf32>
    return %7 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x28x28x192xf32>
    return %1 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x192xf32>
    return %1 : tensor<7x7x1x192xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<784x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x28x28x192xf32>, %arg3: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<1x28x28x192xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<784x192xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<784x192xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<1x28x28x192xf32>
    %5 = stablehlo.add %1, %4 : tensor<1x28x28x192xf32>
    return %5 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<1x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<1x1x1x768xf32>, %arg3: tensor<1x1x1x768xf32>) -> tensor<784x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x28x28x768xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x28x28x768xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x28x28x768xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x28x28x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
    return %10 : tensor<784x768xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x768xf32>) -> tensor<1x1x1x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x768xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x768xf32>
    return %5 : tensor<1x1x1x768xf32>
  }
  func.func private @region_16.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.182(%arg0: tensor<1x28x28x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x28x28x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x28x28x768xf32>, tensor<f32>) -> tensor<1x768xf32>
    %2 = stablehlo.sqrt %1 : tensor<1x768xf32>
    return %2 : tensor<1x768xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<784x768xf32>, %arg1: tensor<768xf32>) -> tensor<1x28x28x768xf32> {
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
  func.func private @fused_computation.184(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<1x28x28xf32>, %arg3: tensor<1x28x28xf32>, %arg4: tensor<1x28x28x192xf32>, %arg5: tensor<192xf32>) -> tensor<784x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x28x28xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x28x28x192xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x28x28x192xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x28x28x192xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x192xf32>) -> tensor<784x192xf32>
    return %14 : tensor<784x192xf32>
  }
  func.func private @region_15.18.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.185(%arg0: tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x28x28xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x28x28xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x28x28xf32>
    return %5 : tensor<1x28x28xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<1x28x28xf32>, %arg1: tensor<1x28x28x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x28x28xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x28x28x192xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x28x28x192xf32>
    return %7 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x28x28x192xf32>
    return %1 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x192xf32>
    return %1 : tensor<7x7x1x192xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<1x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x28x28x192xf32>
    return %1 : tensor<1x28x28x192xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,96,192]{1,0,2,3}"} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
    %1 = mhlo.copy %0 : tensor<2x2x96x192xf32>
    return %1 : tensor<2x2x96x192xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x56x56xf32>, %arg3: tensor<1x56x56x96xf32>, %arg4: tensor<1x56x56xf32>) -> tensor<1x56x56x96xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x56x56xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x56x56x96xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x56x56x96xf32>
    return %11 : tensor<1x56x56x96xf32>
  }
  func.func private @region_2.3.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.192(%arg0: tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x56x56xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x56x56xf32>
    return %5 : tensor<1x56x56xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<1x56x56xf32>) -> tensor<1x56x56x96xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x56x56xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x56x56x96xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x56x56x96xf32>
    return %5 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<3136x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<96xf32>) -> tensor<3136x96xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<3136x96xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<1x56x56x96xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x56x56x96xf32>
    return %3 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.195(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<1x1x1x384xf32>, %arg3: tensor<1x1x1x384xf32>) -> tensor<3136x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x56x56x384xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x56x56x384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x56x56x384xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x56x56x384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    return %10 : tensor<3136x384xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x384xf32>) -> tensor<1x1x1x384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x384xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x384xf32>
    return %5 : tensor<1x1x1x384xf32>
  }
  func.func private @region_10.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.197(%arg0: tensor<1x2x2x384xf32>) -> tensor<1x384xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x2x2x384xf32>, tensor<f32>) -> tensor<1x384xf32>
    %1 = stablehlo.sqrt %0 : tensor<1x384xf32>
    return %1 : tensor<1x384xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<3136x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x56x56x384xf32> {
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
  func.func private @fused_computation.199(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x56x56xf32>, %arg3: tensor<1x56x56xf32>, %arg4: tensor<1x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<3136x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x56x56x96xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x56x56x96xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x96xf32>) -> tensor<3136x96xf32>
    return %14 : tensor<3136x96xf32>
  }
  func.func private @region_2.3.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.200(%arg0: tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x56x56xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x56x56xf32>
    return %5 : tensor<1x56x56xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<1x56x56xf32>, %arg1: tensor<1x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x56x56x96xf32>
    return %7 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x56x56x96xf32>
    return %1 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x96xf32>
    return %1 : tensor<7x7x1x96xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<3136x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<96xf32>) -> tensor<3136x96xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<3136x96xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<1x56x56x96xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<1x56x56x96xf32>
    return %3 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<1x1x1x384xf32>, %arg3: tensor<1x1x1x384xf32>) -> tensor<3136x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x56x56x384xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x56x56x384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x56x56x384xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x56x56x384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    return %10 : tensor<3136x384xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x384xf32>) -> tensor<1x1x1x384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x384xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x384xf32>
    return %5 : tensor<1x1x1x384xf32>
  }
  func.func private @region_7.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.207(%arg0: tensor<1x2x2x384xf32>) -> tensor<1x384xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x2x2x384xf32>, tensor<f32>) -> tensor<1x384xf32>
    %1 = stablehlo.sqrt %0 : tensor<1x384xf32>
    return %1 : tensor<1x384xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<3136x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x56x56x384xf32> {
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
  func.func private @fused_computation.209(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x56x56xf32>, %arg3: tensor<1x56x56xf32>, %arg4: tensor<1x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<3136x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x56x56x96xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x56x56x96xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x96xf32>) -> tensor<3136x96xf32>
    return %14 : tensor<3136x96xf32>
  }
  func.func private @region_2.3.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.210(%arg0: tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x56x56xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x56x56xf32>
    return %5 : tensor<1x56x56xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<1x56x56xf32>, %arg1: tensor<1x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x56x56x96xf32>
    return %7 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x56x56x96xf32>
    return %1 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x96xf32>
    return %1 : tensor<7x7x1x96xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<3136x96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<96xf32>, %arg4: tensor<1x56x56xf32>, %arg5: tensor<1x56x56xf32>, %arg6: tensor<1x56x56x96xf32>, %arg7: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg6, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg5, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x56x56x96xf32>
    %12 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x56x56x96xf32>
    %14 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<3136x96xf32>
    %15 = stablehlo.add %14, %arg0 : tensor<3136x96xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<1x56x56x96xf32>
    %17 = stablehlo.add %13, %16 : tensor<1x56x56x96xf32>
    return %17 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<1x1x1x384xf32>, %arg3: tensor<1x1x1x384xf32>) -> tensor<3136x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<1x56x56x384xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<1x56x56x384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %8 = stablehlo.add %5, %7 : tensor<1x56x56x384xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<1x56x56x384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
    return %10 : tensor<3136x384xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x384xf32>) -> tensor<1x1x1x384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<1x1x1xf32>
    %1 = stablehlo.multiply %arg0, %cst : tensor<1x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
    %2 = stablehlo.add %1, %cst_0 : tensor<1x1x1xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1x384xf32>
    %5 = stablehlo.divide %0, %4 : tensor<1x1x1x384xf32>
    return %5 : tensor<1x1x1x384xf32>
  }
  func.func private @region_4.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.217(%arg0: tensor<1x2x2x384xf32>) -> tensor<1x384xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x2x2x384xf32>, tensor<f32>) -> tensor<1x384xf32>
    %1 = stablehlo.sqrt %0 : tensor<1x384xf32>
    return %1 : tensor<1x384xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<3136x384xf32>, %arg1: tensor<384xf32>) -> tensor<1x56x56x384xf32> {
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
  func.func private @fused_computation.219(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x56x56xf32>, %arg3: tensor<1x56x56xf32>, %arg4: tensor<1x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<3136x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x56x56x96xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x56x56x96xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x96xf32>) -> tensor<3136x96xf32>
    return %14 : tensor<3136x96xf32>
  }
  func.func private @region_2.3.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.220(%arg0: tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x56x56xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x56x56xf32>
    return %5 : tensor<1x56x56xf32>
  }
  func.func private @fused_computation.221(%arg0: tensor<1x56x56xf32>, %arg1: tensor<1x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x56x56x96xf32>
    return %7 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x56x56x96xf32>
    return %1 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x96xf32>
    return %1 : tensor<7x7x1x96xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<1x56x56xf32>, %arg3: tensor<1x56x56xf32>, %arg4: tensor<1x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x56x56x96xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %13 = stablehlo.add %11, %12 : tensor<1x56x56x96xf32>
    return %13 : tensor<1x56x56x96xf32>
  }
  func.func private @region_2.3.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.225(%arg0: tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x56x56xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x56x56xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x56x56xf32>
    return %5 : tensor<1x56x56xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<1x56x56xf32>, %arg1: tensor<1x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<1x56x56xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
    %6 = stablehlo.subtract %1, %5 : tensor<1x56x56x96xf32>
    %7 = stablehlo.multiply %6, %6 : tensor<1x56x56x96xf32>
    return %7 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1x56x56x96xf32>
    return %1 : tensor<1x56x56x96xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,3,96]{1,0,2,3}"} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
    %1 = mhlo.copy %0 : tensor<4x4x3x96xf32>
    return %1 : tensor<4x4x3x96xf32>
  }
  func.func private @fused_computation.229(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<1x224x224x3xf32>
    return %1 : tensor<1x224x224x3xf32>
  }
  func.func private @region_2.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    return %0 : tensor<1x56x56x3xf32>
  }
  func.func private @region_2.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    return %0 : tensor<1x56x56x3xf32>
  }
  func.func private @wrapped_multiply_computation(%arg0: tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x56x56x384xf32>
    return %0 : tensor<1x56x56x384xf32>
  }
  func.func private @region_4.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<f32>) -> tensor<1x2x2x384xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
    return %0 : tensor<1x2x2x384xf32>
  }
  func.func private @region_5.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<1x1x1x384xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
    return %0 : tensor<1x1x1x12xf32>
  }
  func.func private @region_5.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<1x1x1x12xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_2.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    return %0 : tensor<1x56x56x3xf32>
  }
  func.func private @wrapped_multiply_computation.1(%arg0: tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x56x56x384xf32>
    return %0 : tensor<1x56x56x384xf32>
  }
  func.func private @region_7.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<f32>) -> tensor<1x2x2x384xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
    return %0 : tensor<1x2x2x384xf32>
  }
  func.func private @region_8.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<1x1x1x384xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
    return %0 : tensor<1x1x1x12xf32>
  }
  func.func private @region_8.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<1x1x1x12xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_2.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    return %0 : tensor<1x56x56x3xf32>
  }
  func.func private @wrapped_multiply_computation.2(%arg0: tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x56x56x384xf32>
    return %0 : tensor<1x56x56x384xf32>
  }
  func.func private @region_10.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<1x56x56x384xf32>, %arg1: tensor<f32>) -> tensor<1x2x2x384xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
    return %0 : tensor<1x2x2x384xf32>
  }
  func.func private @region_11.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<1x1x1x384xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
    return %0 : tensor<1x1x1x12xf32>
  }
  func.func private @region_11.14.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<1x1x1x12xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_2.3.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<1x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<1x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    return %0 : tensor<1x56x56x3xf32>
  }
  func.func private @region_15.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    return %0 : tensor<1x28x28x6xf32>
  }
  func.func private @region_17.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<1x1x1x768xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
    return %0 : tensor<1x1x1x24xf32>
  }
  func.func private @region_17.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<1x1x1x24xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_15.18.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    return %0 : tensor<1x28x28x6xf32>
  }
  func.func private @region_20.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<1x1x1x768xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
    return %0 : tensor<1x1x1x24xf32>
  }
  func.func private @region_20.25.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<1x1x1x24xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_15.18.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    return %0 : tensor<1x28x28x6xf32>
  }
  func.func private @region_23.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<1x1x1x768xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
    return %0 : tensor<1x1x1x24xf32>
  }
  func.func private @region_23.28.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<1x1x1x24xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_15.18.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<1x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<1x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    return %0 : tensor<1x28x28x6xf32>
  }
  func.func private @region_27.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_29.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_29.36.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_29.36.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_32.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_32.39.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_32.39.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_35.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_35.42.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_35.42.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_38.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_38.45.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_38.45.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_41.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_41.48.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_41.48.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.33(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_44.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.34(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_44.51.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.35(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_44.51.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.36(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_47.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.37(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_47.54.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.38(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_47.54.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.39(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_50.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.40(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_50.57.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.41(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_50.57.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.42(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_53.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.43(%arg0: tensor<1x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    return %0 : tensor<1x1x1x48xf32>
  }
  func.func private @region_53.60.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.44(%arg0: tensor<1x1x1x48xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    return %0 : tensor<1x1x1x2xf32>
  }
  func.func private @region_53.60.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<1x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_27.32.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.45(%arg0: tensor<1x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<1x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    return %0 : tensor<1x14x14x12xf32>
  }
  func.func private @region_57.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.46(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x7x7x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
    return %0 : tensor<1x7x7x24xf32>
  }
  func.func private @region_59.68(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.47(%arg0: tensor<1x1x1x3072xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x96xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    return %0 : tensor<1x1x1x96xf32>
  }
  func.func private @region_59.68.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.48(%arg0: tensor<1x1x1x96xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
    return %0 : tensor<1x1x1x3xf32>
  }
  func.func private @region_59.68.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.15(%arg0: tensor<1x1x1x3xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_57.64.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.49(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x7x7x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
    return %0 : tensor<1x7x7x24xf32>
  }
  func.func private @region_62.71(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.50(%arg0: tensor<1x1x1x3072xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x96xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    return %0 : tensor<1x1x1x96xf32>
  }
  func.func private @region_62.71.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.51(%arg0: tensor<1x1x1x96xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
    return %0 : tensor<1x1x1x3xf32>
  }
  func.func private @region_62.71.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.16(%arg0: tensor<1x1x1x3xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_57.64.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.52(%arg0: tensor<1x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<1x7x7x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
    return %0 : tensor<1x7x7x24xf32>
  }
  func.func private @region_65.74(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.53(%arg0: tensor<1x1x1x3072xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x96xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    return %0 : tensor<1x1x1x96xf32>
  }
  func.func private @region_65.74.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.54(%arg0: tensor<1x1x1x96xf32>, %arg1: tensor<f32>) -> tensor<1x1x1x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
    return %0 : tensor<1x1x1x3xf32>
  }
  func.func private @region_65.74.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.17(%arg0: tensor<1x1x1x3xf32>, %arg1: tensor<f32>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
  func.func private @region_67.76(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.55(%arg0: tensor<1x768xf32>, %arg1: tensor<f32>) -> tensor<1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x768xf32>, tensor<f32>) -> tensor<1x24xf32>
    return %0 : tensor<1x24xf32>
  }
  func.func private @region_67.76.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.18(%arg0: tensor<1x24xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<1x24xf32>, tensor<f32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  func.func private @region_69.78(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.56(%arg0: tensor<1x768xf32>, %arg1: tensor<f32>) -> tensor<1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x768xf32>, tensor<f32>) -> tensor<1x24xf32>
    return %0 : tensor<1x24xf32>
  }
  func.func @main(%arg0: tensor<96x3x4x4xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<96xf32>, %arg4: tensor<96x1x7x7xf32>, %arg5: tensor<96xf32>, %arg6: tensor<96xf32>, %arg7: tensor<96xf32>, %arg8: tensor<384x96xf32>, %arg9: tensor<384xf32>, %arg10: tensor<1x1x1x384xf32>, %arg11: tensor<1x1x1x384xf32>, %arg12: tensor<96x384xf32>, %arg13: tensor<96xf32>, %arg14: tensor<96x1x7x7xf32>, %arg15: tensor<96xf32>, %arg16: tensor<96xf32>, %arg17: tensor<96xf32>, %arg18: tensor<384x96xf32>, %arg19: tensor<384xf32>, %arg20: tensor<1x1x1x384xf32>, %arg21: tensor<1x1x1x384xf32>, %arg22: tensor<96x384xf32>, %arg23: tensor<96xf32>, %arg24: tensor<96x1x7x7xf32>, %arg25: tensor<96xf32>, %arg26: tensor<96xf32>, %arg27: tensor<96xf32>, %arg28: tensor<384x96xf32>, %arg29: tensor<384xf32>, %arg30: tensor<1x1x1x384xf32>, %arg31: tensor<1x1x1x384xf32>, %arg32: tensor<96x384xf32>, %arg33: tensor<96xf32>, %arg34: tensor<96xf32>, %arg35: tensor<96xf32>, %arg36: tensor<192x96x2x2xf32>, %arg37: tensor<192xf32>, %arg38: tensor<192x1x7x7xf32>, %arg39: tensor<192xf32>, %arg40: tensor<192xf32>, %arg41: tensor<192xf32>, %arg42: tensor<768x192xf32>, %arg43: tensor<768xf32>, %arg44: tensor<1x1x1x768xf32>, %arg45: tensor<1x1x1x768xf32>, %arg46: tensor<192x768xf32>, %arg47: tensor<192xf32>, %arg48: tensor<192x1x7x7xf32>, %arg49: tensor<192xf32>, %arg50: tensor<192xf32>, %arg51: tensor<192xf32>, %arg52: tensor<768x192xf32>, %arg53: tensor<768xf32>, %arg54: tensor<1x1x1x768xf32>, %arg55: tensor<1x1x1x768xf32>, %arg56: tensor<192x768xf32>, %arg57: tensor<192xf32>, %arg58: tensor<192x1x7x7xf32>, %arg59: tensor<192xf32>, %arg60: tensor<192xf32>, %arg61: tensor<192xf32>, %arg62: tensor<768x192xf32>, %arg63: tensor<768xf32>, %arg64: tensor<1x1x1x768xf32>, %arg65: tensor<1x1x1x768xf32>, %arg66: tensor<192x768xf32>, %arg67: tensor<192xf32>, %arg68: tensor<192xf32>, %arg69: tensor<192xf32>, %arg70: tensor<384x192x2x2xf32>, %arg71: tensor<384xf32>, %arg72: tensor<384x1x7x7xf32>, %arg73: tensor<384xf32>, %arg74: tensor<384xf32>, %arg75: tensor<384xf32>, %arg76: tensor<1536x384xf32>, %arg77: tensor<1536xf32>, %arg78: tensor<1x1x1x1536xf32>, %arg79: tensor<1x1x1x1536xf32>, %arg80: tensor<384x1536xf32>, %arg81: tensor<384xf32>, %arg82: tensor<384x1x7x7xf32>, %arg83: tensor<384xf32>, %arg84: tensor<384xf32>, %arg85: tensor<384xf32>, %arg86: tensor<1536x384xf32>, %arg87: tensor<1536xf32>, %arg88: tensor<1x1x1x1536xf32>, %arg89: tensor<1x1x1x1536xf32>, %arg90: tensor<384x1536xf32>, %arg91: tensor<384xf32>, %arg92: tensor<384x1x7x7xf32>, %arg93: tensor<384xf32>, %arg94: tensor<384xf32>, %arg95: tensor<384xf32>, %arg96: tensor<1536x384xf32>, %arg97: tensor<1536xf32>, %arg98: tensor<1x1x1x1536xf32>, %arg99: tensor<1x1x1x1536xf32>, %arg100: tensor<384x1536xf32>, %arg101: tensor<384xf32>, %arg102: tensor<384x1x7x7xf32>, %arg103: tensor<384xf32>, %arg104: tensor<384xf32>, %arg105: tensor<384xf32>, %arg106: tensor<1536x384xf32>, %arg107: tensor<1536xf32>, %arg108: tensor<1x1x1x1536xf32>, %arg109: tensor<1x1x1x1536xf32>, %arg110: tensor<384x1536xf32>, %arg111: tensor<384xf32>, %arg112: tensor<384x1x7x7xf32>, %arg113: tensor<384xf32>, %arg114: tensor<384xf32>, %arg115: tensor<384xf32>, %arg116: tensor<1536x384xf32>, %arg117: tensor<1536xf32>, %arg118: tensor<1x1x1x1536xf32>, %arg119: tensor<1x1x1x1536xf32>, %arg120: tensor<384x1536xf32>, %arg121: tensor<384xf32>, %arg122: tensor<384x1x7x7xf32>, %arg123: tensor<384xf32>, %arg124: tensor<384xf32>, %arg125: tensor<384xf32>, %arg126: tensor<1536x384xf32>, %arg127: tensor<1536xf32>, %arg128: tensor<1x1x1x1536xf32>, %arg129: tensor<1x1x1x1536xf32>, %arg130: tensor<384x1536xf32>, %arg131: tensor<384xf32>, %arg132: tensor<384x1x7x7xf32>, %arg133: tensor<384xf32>, %arg134: tensor<384xf32>, %arg135: tensor<384xf32>, %arg136: tensor<1536x384xf32>, %arg137: tensor<1536xf32>, %arg138: tensor<1x1x1x1536xf32>, %arg139: tensor<1x1x1x1536xf32>, %arg140: tensor<384x1536xf32>, %arg141: tensor<384xf32>, %arg142: tensor<384x1x7x7xf32>, %arg143: tensor<384xf32>, %arg144: tensor<384xf32>, %arg145: tensor<384xf32>, %arg146: tensor<1536x384xf32>, %arg147: tensor<1536xf32>, %arg148: tensor<1x1x1x1536xf32>, %arg149: tensor<1x1x1x1536xf32>, %arg150: tensor<384x1536xf32>, %arg151: tensor<384xf32>, %arg152: tensor<384x1x7x7xf32>, %arg153: tensor<384xf32>, %arg154: tensor<384xf32>, %arg155: tensor<384xf32>, %arg156: tensor<1536x384xf32>, %arg157: tensor<1536xf32>, %arg158: tensor<1x1x1x1536xf32>, %arg159: tensor<1x1x1x1536xf32>, %arg160: tensor<384x1536xf32>, %arg161: tensor<384xf32>, %arg162: tensor<384xf32>, %arg163: tensor<384xf32>, %arg164: tensor<768x384x2x2xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x1x7x7xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<3072x768xf32>, %arg171: tensor<3072xf32>, %arg172: tensor<1x1x1x3072xf32>, %arg173: tensor<1x1x1x3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768x1x7x7xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<3072x768xf32>, %arg181: tensor<3072xf32>, %arg182: tensor<1x1x1x3072xf32>, %arg183: tensor<1x1x1x3072xf32>, %arg184: tensor<768x3072xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x1x7x7xf32>, %arg187: tensor<768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<3072x768xf32>, %arg191: tensor<3072xf32>, %arg192: tensor<1x1x1x3072xf32>, %arg193: tensor<1x1x1x3072xf32>, %arg194: tensor<768x3072xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x3x224x224xf32>):
      %386 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
      %387 = mhlo.copy %386 : tensor<1x224x224x3xf32>
      mhlo.return %387 : tensor<1x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x3x4x4xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,3,96]{1,0,2,3}"} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
      %387 = mhlo.copy %386 : tensor<4x4x3x96xf32>
      mhlo.return %387 : tensor<4x4x3x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<4x4x3x96xf32>) -> tensor<1x56x56x96xf32>
    %3 = "mhlo.fusion"(%2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x56x56x96xf32>
      mhlo.return %387 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.fusion"(%3, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      mhlo.return %386 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %5 = "mhlo.fusion"(%4, %2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56xf32>, %arg202: tensor<1x56x56x96xf32>, %arg203: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x56x56x96xf32>
      mhlo.return %393 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
      mhlo.return %386 : tensor<1x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    %7 = "mhlo.fusion"(%6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x3xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x56x56xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x56x56xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x56x56xf32>
      mhlo.return %391 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32>
    %8 = "mhlo.fusion"(%arg3, %arg2, %7, %4, %2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x56x56xf32>, %arg204: tensor<1x56x56xf32>, %arg205: tensor<1x56x56x96xf32>, %arg206: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x56x56x96xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x56x56x96xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x56x56x96xf32>
      mhlo.return %399 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x56x56xf32>, tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %9 = "mhlo.fusion"(%arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x96xf32>
      mhlo.return %387 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %10 = stablehlo.convolution(%8, %9) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<1x56x56x96xf32>
    %11 = "mhlo.fusion"(%10, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x56x56x96xf32>
      mhlo.return %387 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %12 = "mhlo.fusion"(%11, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      mhlo.return %386 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %13 = "mhlo.fusion"(%12, %10, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56xf32>, %arg202: tensor<1x56x56x96xf32>, %arg203: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x56x56x96xf32>
      mhlo.return %393 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %14 = "mhlo.fusion"(%13, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
      mhlo.return %386 : tensor<1x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    %15 = "mhlo.fusion"(%14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x3xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x56x56xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x56x56xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x56x56xf32>
      mhlo.return %391 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32>
    %16 = "mhlo.fusion"(%arg7, %arg6, %15, %12, %10, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x56x56xf32>, %arg204: tensor<1x56x56xf32>, %arg205: tensor<1x56x56x96xf32>, %arg206: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x56x56x96xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x56x56x96xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x56x56x96xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x96xf32>) -> tensor<3136x96xf32>
      mhlo.return %400 : tensor<3136x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x56x56xf32>, tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<3136x96xf32>
    %17 = stablehlo.dot_general %16, %arg8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    %18 = "mhlo.fusion"(%17, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<3136x384xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<3136x384xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %391 = stablehlo.negate %388 : tensor<3136x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<3136x384xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %395 = stablehlo.abs %394 : tensor<1x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x56x56x384xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x56x56x384xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x56x56x384xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x56x56x384xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x56x56x384xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x56x56x384xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x56x56x384xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x56x56x384xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %423 = stablehlo.negate %399 : tensor<1x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %426 = stablehlo.exponential %423 : tensor<1x56x56x384xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x56x56x384xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x56x56x384xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x56x56x384xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x56x56x384xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x56x56x384xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x56x56x384xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x56x56x384xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x56x56x384xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x56x56x384xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x56x56x384xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x56x56x384xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x56x56x384xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x56x56x384xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x56x56x384xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x56x56x384xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x56x56x384xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x56x56x384xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x56x56x384xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x56x56x384xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x56x56x384xf32>
      mhlo.return %484 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x384xf32>, tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %19 = "mhlo.fusion"(%18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x56x56x384xf32>
      mhlo.return %386 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xf32>
    %20 = "mhlo.fusion"(%19, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
      mhlo.return %386 : tensor<1x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
    %21 = "mhlo.fusion"(%20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x2x2x384xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x2x2x384xf32>, tensor<f32>) -> tensor<1x384xf32>
      %387 = stablehlo.sqrt %386 : tensor<1x384xf32>
      mhlo.return %387 : tensor<1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x2x384xf32>) -> tensor<1x384xf32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %23 = "mhlo.fusion"(%22, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
      mhlo.return %386 : tensor<1x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
    %24 = "mhlo.fusion"(%23, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x12xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %25 = "mhlo.fusion"(%24, %21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x384xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x384xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x384xf32>
      mhlo.return %391 : tensor<1x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %26 = "mhlo.fusion"(%18, %arg11, %25, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<1x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x56x56x384xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x56x56x384xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x56x56x384xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x56x56x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
      mhlo.return %396 : tensor<3136x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<3136x384xf32>
    %27 = stablehlo.dot_general %26, %arg12, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    %28 = "mhlo.fusion"(%27, %arg13, %arg3, %arg2, %7, %4, %2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<96xf32>, %arg204: tensor<96xf32>, %arg205: tensor<1x56x56xf32>, %arg206: tensor<1x56x56xf32>, %arg207: tensor<1x56x56x96xf32>, %arg208: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg207, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg206, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = mhlo.bitcast %arg205 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x56x56x96xf32>
      %396 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x56x56x96xf32>
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x56x56x96xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<3136x96xf32>
      %401 = stablehlo.add %400, %arg201 : tensor<3136x96xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<1x56x56x96xf32>
      %403 = stablehlo.add %399, %402 : tensor<1x56x56x96xf32>
      mhlo.return %403 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<1x56x56xf32>, tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %29 = "mhlo.fusion"(%arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x96xf32>
      mhlo.return %387 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %30 = stablehlo.convolution(%28, %29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<1x56x56x96xf32>
    %31 = "mhlo.fusion"(%30, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x56x56x96xf32>
      mhlo.return %387 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %32 = "mhlo.fusion"(%31, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      mhlo.return %386 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %33 = "mhlo.fusion"(%32, %30, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56xf32>, %arg202: tensor<1x56x56x96xf32>, %arg203: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x56x56x96xf32>
      mhlo.return %393 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %34 = "mhlo.fusion"(%33, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
      mhlo.return %386 : tensor<1x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    %35 = "mhlo.fusion"(%34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x3xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x56x56xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x56x56xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x56x56xf32>
      mhlo.return %391 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32>
    %36 = "mhlo.fusion"(%arg17, %arg16, %35, %32, %30, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x56x56xf32>, %arg204: tensor<1x56x56xf32>, %arg205: tensor<1x56x56x96xf32>, %arg206: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x56x56x96xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x56x56x96xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x56x56x96xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x96xf32>) -> tensor<3136x96xf32>
      mhlo.return %400 : tensor<3136x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x56x56xf32>, tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<3136x96xf32>
    %37 = stablehlo.dot_general %36, %arg18, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    %38 = "mhlo.fusion"(%37, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<3136x384xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<3136x384xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %391 = stablehlo.negate %388 : tensor<3136x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<3136x384xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %395 = stablehlo.abs %394 : tensor<1x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x56x56x384xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x56x56x384xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x56x56x384xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x56x56x384xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x56x56x384xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x56x56x384xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x56x56x384xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x56x56x384xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %423 = stablehlo.negate %399 : tensor<1x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %426 = stablehlo.exponential %423 : tensor<1x56x56x384xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x56x56x384xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x56x56x384xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x56x56x384xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x56x56x384xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x56x56x384xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x56x56x384xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x56x56x384xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x56x56x384xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x56x56x384xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x56x56x384xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x56x56x384xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x56x56x384xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x56x56x384xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x56x56x384xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x56x56x384xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x56x56x384xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x56x56x384xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x56x56x384xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x56x56x384xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x56x56x384xf32>
      mhlo.return %484 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x384xf32>, tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %39 = "mhlo.fusion"(%38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x56x56x384xf32>
      mhlo.return %386 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xf32>
    %40 = "mhlo.fusion"(%39, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
      mhlo.return %386 : tensor<1x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x2x2x384xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x2x2x384xf32>, tensor<f32>) -> tensor<1x384xf32>
      %387 = stablehlo.sqrt %386 : tensor<1x384xf32>
      mhlo.return %387 : tensor<1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x2x384xf32>) -> tensor<1x384xf32>
    %42 = mhlo.bitcast %41 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %43 = "mhlo.fusion"(%42, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
      mhlo.return %386 : tensor<1x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
    %44 = "mhlo.fusion"(%43, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x12xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %45 = "mhlo.fusion"(%44, %41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x384xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x384xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x384xf32>
      mhlo.return %391 : tensor<1x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %46 = "mhlo.fusion"(%38, %arg21, %45, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<1x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x56x56x384xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x56x56x384xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x56x56x384xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x56x56x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
      mhlo.return %396 : tensor<3136x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<3136x384xf32>
    %47 = stablehlo.dot_general %46, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    %48 = "mhlo.fusion"(%28, %47, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<3136x96xf32>, %arg203: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<96xf32>) -> tensor<3136x96xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<3136x96xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<1x56x56x96xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x56x56x96xf32>
      mhlo.return %389 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<3136x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %49 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x96xf32>
      mhlo.return %387 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %50 = stablehlo.convolution(%48, %49) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<1x56x56x96xf32>
    %51 = "mhlo.fusion"(%50, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x56x56x96xf32>
      mhlo.return %387 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %52 = "mhlo.fusion"(%51, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      mhlo.return %386 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %53 = "mhlo.fusion"(%52, %50, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56xf32>, %arg202: tensor<1x56x56x96xf32>, %arg203: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x56x56x96xf32>
      mhlo.return %393 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %54 = "mhlo.fusion"(%53, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
      mhlo.return %386 : tensor<1x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    %55 = "mhlo.fusion"(%54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x3xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x56x56xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x56x56xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x56x56xf32>
      mhlo.return %391 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32>
    %56 = "mhlo.fusion"(%arg27, %arg26, %55, %52, %50, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x56x56xf32>, %arg204: tensor<1x56x56xf32>, %arg205: tensor<1x56x56x96xf32>, %arg206: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x56x56xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x56x56x96xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x56x56x96xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x56x56x96xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x56x56x96xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x96xf32>) -> tensor<3136x96xf32>
      mhlo.return %400 : tensor<3136x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x56x56xf32>, tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<96xf32>) -> tensor<3136x96xf32>
    %57 = stablehlo.dot_general %56, %arg28, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x96xf32>, tensor<384x96xf32>) -> tensor<3136x384xf32>
    %58 = "mhlo.fusion"(%57, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<3136x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<3136x384xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<3136x384xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<3136x384xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %391 = stablehlo.negate %388 : tensor<3136x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3136x384xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<3136x384xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x384xf32>) -> tensor<1x56x56x384xf32>
      %395 = stablehlo.abs %394 : tensor<1x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x56x56x384xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x56x56x384xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x56x56x384xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x56x56x384xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x56x56x384xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x56x56x384xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x56x56x384xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x56x56x384xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %423 = stablehlo.negate %399 : tensor<1x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %426 = stablehlo.exponential %423 : tensor<1x56x56x384xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x56x56x384xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x56x56x384xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x56x56x384xf32>, tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x56x56x384xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x56x56x384xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x56x56x384xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x56x56x384xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x56x56x384xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x56x56x384xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x56x56x384xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x56x56x384xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x56x56x384xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x56x56x384xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x56x56x384xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x56x56x384xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x56x56x384xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x56x56x384xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x56x56x384xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x56x56x384xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x56x56x384xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x56x56x384xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x56x56x384xi1>, tensor<1x56x56x384xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x56x56x384xf32>
      mhlo.return %484 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<3136x384xf32>, tensor<384xf32>) -> tensor<1x56x56x384xf32>
    %59 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x56x56x384xf32>
      mhlo.return %386 : tensor<1x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>) -> tensor<1x56x56x384xf32>
    %60 = "mhlo.fusion"(%59, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
      mhlo.return %386 : tensor<1x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<f32>) -> tensor<1x2x2x384xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x2x2x384xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x2x2x384xf32>, tensor<f32>) -> tensor<1x384xf32>
      %387 = stablehlo.sqrt %386 : tensor<1x384xf32>
      mhlo.return %387 : tensor<1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x2x2x384xf32>) -> tensor<1x384xf32>
    %62 = mhlo.bitcast %61 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %63 = "mhlo.fusion"(%62, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
      mhlo.return %386 : tensor<1x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x384xf32>, tensor<f32>) -> tensor<1x1x1x12xf32>
    %64 = "mhlo.fusion"(%63, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x12xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x12xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %65 = "mhlo.fusion"(%64, %61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x384xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x384xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x384xf32>
      mhlo.return %391 : tensor<1x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x384xf32>) -> tensor<1x1x1x384xf32>
    %66 = "mhlo.fusion"(%58, %arg31, %65, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<1x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x56x56x384xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x56x56x384xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<384xf32>) -> tensor<1x56x56x384xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x56x56x384xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x56x56x384xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x56x56x384xf32>) -> tensor<3136x384xf32>
      mhlo.return %396 : tensor<3136x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<3136x384xf32>
    %67 = stablehlo.dot_general %66, %arg32, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3136x384xf32>, tensor<96x384xf32>) -> tensor<3136x96xf32>
    %68 = "mhlo.fusion"(%48, %67, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<3136x96xf32>, %arg203: tensor<96xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<96xf32>) -> tensor<3136x96xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<3136x96xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<3136x96xf32>) -> tensor<1x56x56x96xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x56x56x96xf32>
      mhlo.return %389 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<3136x96xf32>, tensor<96xf32>) -> tensor<1x56x56x96xf32>
    %69 = "mhlo.fusion"(%68, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      mhlo.return %386 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56xf32>
    %70 = "mhlo.fusion"(%68, %69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<1x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %387 = stablehlo.multiply %arg202, %386 : tensor<1x56x56xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %390 = stablehlo.subtract %arg201, %389 : tensor<1x56x56x96xf32>
      %391 = stablehlo.multiply %390, %390 : tensor<1x56x56x96xf32>
      mhlo.return %391 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<1x56x56xf32>) -> tensor<1x56x56x96xf32>
    %71 = "mhlo.fusion"(%70, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
      mhlo.return %386 : tensor<1x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x96xf32>, tensor<f32>) -> tensor<1x56x56x3xf32>
    %72 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x56x56x3xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x56x56x3xf32>, tensor<f32>) -> tensor<1x56x56xf32>
      %cst_1 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x56x56xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x56x56xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x56x56xf32>
      mhlo.return %391 : tensor<1x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<1x56x56x3xf32>) -> tensor<1x56x56xf32>
    %73 = "mhlo.fusion"(%arg35, %arg34, %72, %68, %69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<1x56x56xf32>, %arg204: tensor<1x56x56x96xf32>, %arg205: tensor<1x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x56x56xf32>
      %387 = stablehlo.multiply %arg205, %386 : tensor<1x56x56xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %390 = stablehlo.subtract %arg204, %389 : tensor<1x56x56x96xf32>
      %391 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x56x56xf32>) -> tensor<56x56xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<56x56xf32>) -> tensor<1x56x56x96xf32>
      %393 = stablehlo.multiply %390, %392 : tensor<1x56x56x96xf32>
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %395 = stablehlo.multiply %393, %394 : tensor<1x56x56x96xf32>
      %396 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<1x56x56x96xf32>
      %397 = stablehlo.add %395, %396 : tensor<1x56x56x96xf32>
      mhlo.return %397 : tensor<1x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<1x56x56xf32>, tensor<1x56x56x96xf32>, tensor<1x56x56xf32>) -> tensor<1x56x56x96xf32>
    %74 = "mhlo.fusion"(%arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x96x2x2xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,96,192]{1,0,2,3}"} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
      %387 = mhlo.copy %386 : tensor<2x2x96x192xf32>
      mhlo.return %387 : tensor<2x2x96x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
    %75 = stablehlo.convolution(%73, %74) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x96xf32>, tensor<2x2x96x192xf32>) -> tensor<1x28x28x192xf32>
    %76 = "mhlo.fusion"(%75, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x28x28x192xf32>
      mhlo.return %387 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %77 = "mhlo.fusion"(%arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x192xf32>
      mhlo.return %387 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %78 = stablehlo.convolution(%76, %77) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<1x28x28x192xf32>
    %79 = "mhlo.fusion"(%78, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x28x28x192xf32>
      mhlo.return %387 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %80 = "mhlo.fusion"(%79, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      mhlo.return %386 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %81 = "mhlo.fusion"(%80, %78, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28xf32>, %arg202: tensor<1x28x28x192xf32>, %arg203: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x28x28xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x28x28x192xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x28x28x192xf32>
      mhlo.return %393 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %82 = "mhlo.fusion"(%81, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
      mhlo.return %386 : tensor<1x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x6xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      %cst_1 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x28x28xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x28x28xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x28x28xf32>
      mhlo.return %391 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32>
    %84 = "mhlo.fusion"(%arg41, %arg40, %83, %80, %78, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<1x28x28xf32>, %arg204: tensor<1x28x28xf32>, %arg205: tensor<1x28x28x192xf32>, %arg206: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x28x28xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x28x28x192xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x28x28x192xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x28x28x192xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x28x28x192xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x192xf32>) -> tensor<784x192xf32>
      mhlo.return %400 : tensor<784x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x28x28xf32>, tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<784x192xf32>
    %85 = stablehlo.dot_general %84, %arg42, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    %86 = "mhlo.fusion"(%85, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<784x768xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<784x768xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %391 = stablehlo.negate %388 : tensor<784x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<784x768xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %395 = stablehlo.abs %394 : tensor<1x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x28x28x768xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x28x28x768xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x28x28x768xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x28x28x768xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x28x28x768xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x28x28x768xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x28x28x768xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x28x28x768xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %423 = stablehlo.negate %399 : tensor<1x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %426 = stablehlo.exponential %423 : tensor<1x28x28x768xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x28x28x768xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x28x28x768xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x28x28x768xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x28x28x768xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x28x28x768xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x28x28x768xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x28x28x768xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x28x28x768xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x28x28x768xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x28x28x768xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x28x28x768xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x28x28x768xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x28x28x768xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x28x28x768xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x28x28x768xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x28x28x768xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x28x28x768xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x28x28x768xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x28x28x768xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x28x28x768xf32>
      mhlo.return %484 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x768xf32>, tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x28x28x768xf32>, tensor<f32>) -> tensor<1x768xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x768xf32>
      mhlo.return %388 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>) -> tensor<1x768xf32>
    %88 = mhlo.bitcast %87 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %89 = "mhlo.fusion"(%88, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
      mhlo.return %386 : tensor<1x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
    %90 = "mhlo.fusion"(%89, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x24xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %91 = "mhlo.fusion"(%90, %87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x768xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x768xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x768xf32>
      mhlo.return %391 : tensor<1x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %92 = "mhlo.fusion"(%86, %arg45, %91, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<1x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x28x28x768xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x28x28x768xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x28x28x768xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x28x28x768xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
      mhlo.return %396 : tensor<784x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<784x768xf32>
    %93 = stablehlo.dot_general %92, %arg46, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    %94 = "mhlo.fusion"(%93, %arg47, %75, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<1x28x28x192xf32>, %arg204: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg203, %386 : tensor<1x28x28x192xf32>
      %388 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<784x192xf32>
      %389 = stablehlo.add %388, %arg201 : tensor<784x192xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<1x28x28x192xf32>
      %391 = stablehlo.add %387, %390 : tensor<1x28x28x192xf32>
      mhlo.return %391 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<784x192xf32>, tensor<192xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %95 = "mhlo.fusion"(%arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x192xf32>
      mhlo.return %387 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %96 = stablehlo.convolution(%94, %95) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<1x28x28x192xf32>
    %97 = "mhlo.fusion"(%96, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x28x28x192xf32>
      mhlo.return %387 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %98 = "mhlo.fusion"(%97, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      mhlo.return %386 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %99 = "mhlo.fusion"(%98, %96, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28xf32>, %arg202: tensor<1x28x28x192xf32>, %arg203: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x28x28xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x28x28x192xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x28x28x192xf32>
      mhlo.return %393 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %100 = "mhlo.fusion"(%99, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
      mhlo.return %386 : tensor<1x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    %101 = "mhlo.fusion"(%100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x6xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      %cst_1 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x28x28xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x28x28xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x28x28xf32>
      mhlo.return %391 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32>
    %102 = "mhlo.fusion"(%arg51, %arg50, %101, %98, %96, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<1x28x28xf32>, %arg204: tensor<1x28x28xf32>, %arg205: tensor<1x28x28x192xf32>, %arg206: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x28x28xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x28x28x192xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x28x28x192xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x28x28x192xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x28x28x192xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x192xf32>) -> tensor<784x192xf32>
      mhlo.return %400 : tensor<784x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x28x28xf32>, tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<784x192xf32>
    %103 = stablehlo.dot_general %102, %arg52, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    %104 = "mhlo.fusion"(%103, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<784x768xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<784x768xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %391 = stablehlo.negate %388 : tensor<784x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<784x768xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %395 = stablehlo.abs %394 : tensor<1x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x28x28x768xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x28x28x768xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x28x28x768xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x28x28x768xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x28x28x768xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x28x28x768xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x28x28x768xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x28x28x768xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %423 = stablehlo.negate %399 : tensor<1x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %426 = stablehlo.exponential %423 : tensor<1x28x28x768xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x28x28x768xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x28x28x768xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x28x28x768xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x28x28x768xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x28x28x768xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x28x28x768xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x28x28x768xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x28x28x768xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x28x28x768xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x28x28x768xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x28x28x768xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x28x28x768xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x28x28x768xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x28x28x768xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x28x28x768xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x28x28x768xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x28x28x768xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x28x28x768xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x28x28x768xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x28x28x768xf32>
      mhlo.return %484 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x768xf32>, tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x28x28x768xf32>, tensor<f32>) -> tensor<1x768xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x768xf32>
      mhlo.return %388 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>) -> tensor<1x768xf32>
    %106 = mhlo.bitcast %105 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
      mhlo.return %386 : tensor<1x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x24xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %109 = "mhlo.fusion"(%108, %105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x768xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x768xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x768xf32>
      mhlo.return %391 : tensor<1x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %110 = "mhlo.fusion"(%104, %arg55, %109, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<1x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x28x28x768xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x28x28x768xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x28x28x768xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x28x28x768xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
      mhlo.return %396 : tensor<784x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<784x768xf32>
    %111 = stablehlo.dot_general %110, %arg56, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    %112 = "mhlo.fusion"(%94, %111, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<784x192xf32>, %arg203: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<192xf32>) -> tensor<784x192xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<784x192xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<1x28x28x192xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x28x28x192xf32>
      mhlo.return %389 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<784x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %113 = "mhlo.fusion"(%arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x192xf32>
      mhlo.return %387 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %114 = stablehlo.convolution(%112, %113) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<1x28x28x192xf32>
    %115 = "mhlo.fusion"(%114, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x28x28x192xf32>
      mhlo.return %387 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %116 = "mhlo.fusion"(%115, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      mhlo.return %386 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %117 = "mhlo.fusion"(%116, %114, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28xf32>, %arg202: tensor<1x28x28x192xf32>, %arg203: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x28x28xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x28x28x192xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x28x28x192xf32>
      mhlo.return %393 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %118 = "mhlo.fusion"(%117, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
      mhlo.return %386 : tensor<1x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x6xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      %cst_1 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x28x28xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x28x28xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x28x28xf32>
      mhlo.return %391 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32>
    %120 = "mhlo.fusion"(%arg61, %arg60, %119, %116, %114, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<1x28x28xf32>, %arg204: tensor<1x28x28xf32>, %arg205: tensor<1x28x28x192xf32>, %arg206: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x28x28xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x28x28x192xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x28x28x192xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x28x28x192xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x28x28x192xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x192xf32>) -> tensor<784x192xf32>
      mhlo.return %400 : tensor<784x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x28x28xf32>, tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<192xf32>) -> tensor<784x192xf32>
    %121 = stablehlo.dot_general %120, %arg62, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x192xf32>, tensor<768x192xf32>) -> tensor<784x768xf32>
    %122 = "mhlo.fusion"(%121, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<784x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<784x768xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<784x768xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<784x768xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %391 = stablehlo.negate %388 : tensor<784x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x768xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<784x768xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x768xf32>) -> tensor<1x28x28x768xf32>
      %395 = stablehlo.abs %394 : tensor<1x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x28x28x768xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x28x28x768xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x28x28x768xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x28x28x768xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x28x28x768xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x28x28x768xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x28x28x768xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x28x28x768xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %423 = stablehlo.negate %399 : tensor<1x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %426 = stablehlo.exponential %423 : tensor<1x28x28x768xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x28x28x768xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x28x28x768xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x28x28x768xf32>, tensor<1x28x28x768xf32>) -> tensor<1x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x28x28x768xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x28x28x768xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x28x28x768xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x28x28x768xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x28x28x768xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x28x28x768xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x28x28x768xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x28x28x768xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x28x28x768xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x28x28x768xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x28x28x768xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x28x28x768xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x28x28x768xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x28x28x768xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x28x28x768xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x28x28x768xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x28x28x768xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x28x28x768xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x28x28x768xi1>, tensor<1x28x28x768xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x28x28x768xf32>
      mhlo.return %484 : tensor<1x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<784x768xf32>, tensor<768xf32>) -> tensor<1x28x28x768xf32>
    %123 = "mhlo.fusion"(%122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x28x28x768xf32>, tensor<f32>) -> tensor<1x768xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x768xf32>
      mhlo.return %388 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>) -> tensor<1x768xf32>
    %124 = mhlo.bitcast %123 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %125 = "mhlo.fusion"(%124, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
      mhlo.return %386 : tensor<1x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x768xf32>, tensor<f32>) -> tensor<1x1x1x24xf32>
    %126 = "mhlo.fusion"(%125, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x24xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x24xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %127 = "mhlo.fusion"(%126, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x768xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x768xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x768xf32>
      mhlo.return %391 : tensor<1x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x768xf32>) -> tensor<1x1x1x768xf32>
    %128 = "mhlo.fusion"(%122, %arg65, %127, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<1x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x28x28x768xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x28x28x768xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<768xf32>) -> tensor<1x28x28x768xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x28x28x768xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x28x28x768xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x28x28x768xf32>) -> tensor<784x768xf32>
      mhlo.return %396 : tensor<784x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<784x768xf32>
    %129 = stablehlo.dot_general %128, %arg66, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<784x768xf32>, tensor<192x768xf32>) -> tensor<784x192xf32>
    %130 = "mhlo.fusion"(%112, %129, %arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<784x192xf32>, %arg203: tensor<192xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<192xf32>) -> tensor<784x192xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<784x192xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<784x192xf32>) -> tensor<1x28x28x192xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x28x28x192xf32>
      mhlo.return %389 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<784x192xf32>, tensor<192xf32>) -> tensor<1x28x28x192xf32>
    %131 = "mhlo.fusion"(%130, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      mhlo.return %386 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28xf32>
    %132 = "mhlo.fusion"(%130, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<1x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %387 = stablehlo.multiply %arg202, %386 : tensor<1x28x28xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %390 = stablehlo.subtract %arg201, %389 : tensor<1x28x28x192xf32>
      %391 = stablehlo.multiply %390, %390 : tensor<1x28x28x192xf32>
      mhlo.return %391 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<1x28x28xf32>) -> tensor<1x28x28x192xf32>
    %133 = "mhlo.fusion"(%132, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x192xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
      mhlo.return %386 : tensor<1x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x192xf32>, tensor<f32>) -> tensor<1x28x28x6xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x28x28x6xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x28x28x6xf32>, tensor<f32>) -> tensor<1x28x28xf32>
      %cst_1 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x28x28xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x28x28xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x28x28xf32>
      mhlo.return %391 : tensor<1x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<1x28x28x6xf32>) -> tensor<1x28x28xf32>
    %135 = "mhlo.fusion"(%arg69, %arg68, %134, %130, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<1x28x28xf32>, %arg204: tensor<1x28x28x192xf32>, %arg205: tensor<1x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x28x28xf32>
      %387 = stablehlo.multiply %arg205, %386 : tensor<1x28x28xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %390 = stablehlo.subtract %arg204, %389 : tensor<1x28x28x192xf32>
      %391 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x28x28xf32>) -> tensor<28x28xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<28x28xf32>) -> tensor<1x28x28x192xf32>
      %393 = stablehlo.multiply %390, %392 : tensor<1x28x28x192xf32>
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %395 = stablehlo.multiply %393, %394 : tensor<1x28x28x192xf32>
      %396 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<1x28x28x192xf32>
      %397 = stablehlo.add %395, %396 : tensor<1x28x28x192xf32>
      mhlo.return %397 : tensor<1x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<1x28x28xf32>, tensor<1x28x28x192xf32>, tensor<1x28x28xf32>) -> tensor<1x28x28x192xf32>
    %136 = "mhlo.fusion"(%arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x192x2x2xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,192,384]{1,0,2,3}"} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
      %387 = mhlo.copy %386 : tensor<2x2x192x384xf32>
      mhlo.return %387 : tensor<2x2x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
    %137 = stablehlo.convolution(%135, %136) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x28x28x192xf32>, tensor<2x2x192x384xf32>) -> tensor<1x14x14x384xf32>
    %138 = "mhlo.fusion"(%137, %arg71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %139 = "mhlo.fusion"(%arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %140 = stablehlo.convolution(%138, %139) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %141 = "mhlo.fusion"(%140, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %143 = "mhlo.fusion"(%142, %140, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %144 = "mhlo.fusion"(%143, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %146 = "mhlo.fusion"(%arg75, %arg74, %145, %142, %140, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %147 = stablehlo.dot_general %146, %arg76, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %148 = "mhlo.fusion"(%147, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %150 = mhlo.bitcast %149 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %151 = "mhlo.fusion"(%150, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %152 = "mhlo.fusion"(%151, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %153 = "mhlo.fusion"(%152, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %154 = "mhlo.fusion"(%153, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %155 = "mhlo.fusion"(%148, %arg79, %154, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %156 = stablehlo.dot_general %155, %arg80, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %157 = "mhlo.fusion"(%156, %arg81, %137, %arg71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14x384xf32>, %arg204: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg203, %386 : tensor<1x14x14x384xf32>
      %388 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %389 = stablehlo.add %388, %arg201 : tensor<196x384xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %391 = stablehlo.add %387, %390 : tensor<1x14x14x384xf32>
      mhlo.return %391 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<196x384xf32>, tensor<384xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %158 = "mhlo.fusion"(%arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %159 = stablehlo.convolution(%157, %158) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %160 = "mhlo.fusion"(%159, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %161 = "mhlo.fusion"(%160, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %162 = "mhlo.fusion"(%161, %159, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %163 = "mhlo.fusion"(%162, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %165 = "mhlo.fusion"(%arg85, %arg84, %164, %161, %159, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %166 = stablehlo.dot_general %165, %arg86, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %167 = "mhlo.fusion"(%166, %arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %168 = "mhlo.fusion"(%167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %169 = mhlo.bitcast %168 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %170 = "mhlo.fusion"(%169, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %171 = "mhlo.fusion"(%170, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %172 = "mhlo.fusion"(%171, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %173 = "mhlo.fusion"(%172, %168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %174 = "mhlo.fusion"(%167, %arg89, %173, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %175 = stablehlo.dot_general %174, %arg90, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %176 = "mhlo.fusion"(%157, %175, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %177 = "mhlo.fusion"(%arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %178 = stablehlo.convolution(%176, %177) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %179 = "mhlo.fusion"(%178, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %180 = "mhlo.fusion"(%179, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %181 = "mhlo.fusion"(%180, %178, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %182 = "mhlo.fusion"(%181, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %184 = "mhlo.fusion"(%arg95, %arg94, %183, %180, %178, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %185 = stablehlo.dot_general %184, %arg96, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %186 = "mhlo.fusion"(%185, %arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %188 = mhlo.bitcast %187 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %189 = "mhlo.fusion"(%188, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %190 = "mhlo.fusion"(%189, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %191 = "mhlo.fusion"(%190, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %192 = "mhlo.fusion"(%191, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %193 = "mhlo.fusion"(%186, %arg99, %192, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %194 = stablehlo.dot_general %193, %arg100, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %195 = "mhlo.fusion"(%176, %194, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %196 = "mhlo.fusion"(%arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %197 = stablehlo.convolution(%195, %196) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %198 = "mhlo.fusion"(%197, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %200 = "mhlo.fusion"(%199, %197, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %201 = "mhlo.fusion"(%200, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %202 = "mhlo.fusion"(%201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %203 = "mhlo.fusion"(%arg105, %arg104, %202, %199, %197, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %204 = stablehlo.dot_general %203, %arg106, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %205 = "mhlo.fusion"(%204, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %206 = "mhlo.fusion"(%205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %207 = mhlo.bitcast %206 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %208 = "mhlo.fusion"(%207, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %209 = "mhlo.fusion"(%208, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %210 = "mhlo.fusion"(%209, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %211 = "mhlo.fusion"(%210, %206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %212 = "mhlo.fusion"(%205, %arg109, %211, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %213 = stablehlo.dot_general %212, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %214 = "mhlo.fusion"(%195, %213, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %215 = "mhlo.fusion"(%arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %216 = stablehlo.convolution(%214, %215) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %217 = "mhlo.fusion"(%216, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %218 = "mhlo.fusion"(%217, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %219 = "mhlo.fusion"(%218, %216, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %220 = "mhlo.fusion"(%219, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %221 = "mhlo.fusion"(%220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %222 = "mhlo.fusion"(%arg115, %arg114, %221, %218, %216, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %223 = stablehlo.dot_general %222, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %224 = "mhlo.fusion"(%223, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %226 = mhlo.bitcast %225 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %227 = "mhlo.fusion"(%226, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %228 = "mhlo.fusion"(%227, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %229 = "mhlo.fusion"(%228, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %230 = "mhlo.fusion"(%229, %225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %231 = "mhlo.fusion"(%224, %arg119, %230, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %232 = stablehlo.dot_general %231, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %233 = "mhlo.fusion"(%214, %232, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %234 = "mhlo.fusion"(%arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %235 = stablehlo.convolution(%233, %234) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %236 = "mhlo.fusion"(%235, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %237 = "mhlo.fusion"(%236, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %238 = "mhlo.fusion"(%237, %235, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %239 = "mhlo.fusion"(%238, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %240 = "mhlo.fusion"(%239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %241 = "mhlo.fusion"(%arg125, %arg124, %240, %237, %235, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %242 = stablehlo.dot_general %241, %arg126, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %243 = "mhlo.fusion"(%242, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %246 = "mhlo.fusion"(%245, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %247 = "mhlo.fusion"(%246, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %248 = "mhlo.fusion"(%247, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %249 = "mhlo.fusion"(%248, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %250 = "mhlo.fusion"(%243, %arg129, %249, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %251 = stablehlo.dot_general %250, %arg130, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %252 = "mhlo.fusion"(%233, %251, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %253 = "mhlo.fusion"(%arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %254 = stablehlo.convolution(%252, %253) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %255 = "mhlo.fusion"(%254, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %257 = "mhlo.fusion"(%256, %254, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %258 = "mhlo.fusion"(%257, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %260 = "mhlo.fusion"(%arg135, %arg134, %259, %256, %254, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %261 = stablehlo.dot_general %260, %arg136, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %262 = "mhlo.fusion"(%261, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %263 = "mhlo.fusion"(%262) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %265 = "mhlo.fusion"(%264, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %266 = "mhlo.fusion"(%265, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %267 = "mhlo.fusion"(%266, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %268 = "mhlo.fusion"(%267, %263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %269 = "mhlo.fusion"(%262, %arg139, %268, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %270 = stablehlo.dot_general %269, %arg140, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %271 = "mhlo.fusion"(%252, %270, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %272 = "mhlo.fusion"(%arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %273 = stablehlo.convolution(%271, %272) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %274 = "mhlo.fusion"(%273, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %275 = "mhlo.fusion"(%274, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %276 = "mhlo.fusion"(%275, %273, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %277 = "mhlo.fusion"(%276, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %278 = "mhlo.fusion"(%277) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %279 = "mhlo.fusion"(%arg145, %arg144, %278, %275, %273, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %280 = stablehlo.dot_general %279, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %281 = "mhlo.fusion"(%280, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %282 = "mhlo.fusion"(%281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %283 = mhlo.bitcast %282 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %284 = "mhlo.fusion"(%283, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %285 = "mhlo.fusion"(%284, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %286 = "mhlo.fusion"(%285, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %287 = "mhlo.fusion"(%286, %282) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %288 = "mhlo.fusion"(%281, %arg149, %287, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %289 = stablehlo.dot_general %288, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %290 = "mhlo.fusion"(%271, %289, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %291 = "mhlo.fusion"(%arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x384xf32>
      mhlo.return %387 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %292 = stablehlo.convolution(%290, %291) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<1x14x14x384xf32>
    %293 = "mhlo.fusion"(%292, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x14x14x384xf32>
      mhlo.return %387 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %294 = "mhlo.fusion"(%293, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %295 = "mhlo.fusion"(%294, %292, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14xf32>, %arg202: tensor<1x14x14x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x14x14x384xf32>
      mhlo.return %393 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %296 = "mhlo.fusion"(%295, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %297 = "mhlo.fusion"(%296) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %298 = "mhlo.fusion"(%arg155, %arg154, %297, %294, %292, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14xf32>, %arg205: tensor<1x14x14x384xf32>, %arg206: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x14x14xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x14x14x384xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x14x14x384xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x14x14x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x384xf32>) -> tensor<196x384xf32>
      mhlo.return %400 : tensor<196x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<384xf32>) -> tensor<196x384xf32>
    %299 = stablehlo.dot_general %298, %arg156, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x384xf32>, tensor<1536x384xf32>) -> tensor<196x1536xf32>
    %300 = "mhlo.fusion"(%299, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<196x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<196x1536xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<196x1536xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<196x1536xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %391 = stablehlo.negate %388 : tensor<196x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<196x1536xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<196x1536xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x1536xf32>) -> tensor<1x14x14x1536xf32>
      %395 = stablehlo.abs %394 : tensor<1x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x14x14x1536xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x14x14x1536xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x14x14x1536xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x14x14x1536xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x14x14x1536xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x14x14x1536xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x14x14x1536xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %423 = stablehlo.negate %399 : tensor<1x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %426 = stablehlo.exponential %423 : tensor<1x14x14x1536xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x14x14x1536xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x14x14x1536xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x14x14x1536xf32>, tensor<1x14x14x1536xf32>) -> tensor<1x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x14x14x1536xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x14x14x1536xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x14x14x1536xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x14x14x1536xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x14x14x1536xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x14x14x1536xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x14x14x1536xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x14x14x1536xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x14x14x1536xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x14x14x1536xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x14x14x1536xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x14x14x1536xi1>, tensor<1x14x14x1536xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x14x14x1536xf32>
      mhlo.return %484 : tensor<1x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<196x1536xf32>, tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
    %301 = "mhlo.fusion"(%300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x14x14x1536xf32>, tensor<f32>) -> tensor<1x1536xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x1536xf32>
      mhlo.return %388 : tensor<1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>) -> tensor<1x1536xf32>
    %302 = mhlo.bitcast %301 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %303 = "mhlo.fusion"(%302, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x1536xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
      mhlo.return %386 : tensor<1x1x1x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x1536xf32>, tensor<f32>) -> tensor<1x1x1x48xf32>
    %304 = "mhlo.fusion"(%303, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x48xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [8, 8]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
      mhlo.return %386 : tensor<1x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x48xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
    %305 = "mhlo.fusion"(%304, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x2xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %306 = "mhlo.fusion"(%305, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x1536xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x1536xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x1536xf32>
      mhlo.return %391 : tensor<1x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x1536xf32>) -> tensor<1x1x1x1536xf32>
    %307 = "mhlo.fusion"(%300, %arg159, %306, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<1x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x14x14x1536xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x14x14x1536xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<1536xf32>) -> tensor<1x14x14x1536xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x14x14x1536xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x14x14x1536xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x14x14x1536xf32>) -> tensor<196x1536xf32>
      mhlo.return %396 : tensor<196x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<196x1536xf32>
    %308 = stablehlo.dot_general %307, %arg160, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<196x1536xf32>, tensor<384x1536xf32>) -> tensor<196x384xf32>
    %309 = "mhlo.fusion"(%290, %308, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<196x384xf32>, %arg203: tensor<384xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<196x384xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<196x384xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<196x384xf32>) -> tensor<1x14x14x384xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x14x14x384xf32>
      mhlo.return %389 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<196x384xf32>, tensor<384xf32>) -> tensor<1x14x14x384xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      mhlo.return %386 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14xf32>
    %311 = "mhlo.fusion"(%309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<1x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %387 = stablehlo.multiply %arg202, %386 : tensor<1x14x14xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %390 = stablehlo.subtract %arg201, %389 : tensor<1x14x14x384xf32>
      %391 = stablehlo.multiply %390, %390 : tensor<1x14x14x384xf32>
      mhlo.return %391 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<1x14x14xf32>) -> tensor<1x14x14x384xf32>
    %312 = "mhlo.fusion"(%311, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x384xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
      mhlo.return %386 : tensor<1x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x384xf32>, tensor<f32>) -> tensor<1x14x14x12xf32>
    %313 = "mhlo.fusion"(%312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x14x14x12xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x14x14x12xf32>, tensor<f32>) -> tensor<1x14x14xf32>
      %cst_1 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x14x14xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x14x14xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x14x14xf32>
      mhlo.return %391 : tensor<1x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<1x14x14x12xf32>) -> tensor<1x14x14xf32>
    %314 = "mhlo.fusion"(%arg163, %arg162, %313, %309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<1x14x14xf32>, %arg204: tensor<1x14x14x384xf32>, %arg205: tensor<1x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x14x14xf32>
      %387 = stablehlo.multiply %arg205, %386 : tensor<1x14x14xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %390 = stablehlo.subtract %arg204, %389 : tensor<1x14x14x384xf32>
      %391 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x14x14xf32>) -> tensor<14x14xf32>
      %392 = stablehlo.broadcast_in_dim %391, dims = [1, 2] : (tensor<14x14xf32>) -> tensor<1x14x14x384xf32>
      %393 = stablehlo.multiply %390, %392 : tensor<1x14x14x384xf32>
      %394 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %395 = stablehlo.multiply %393, %394 : tensor<1x14x14x384xf32>
      %396 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<1x14x14x384xf32>
      %397 = stablehlo.add %395, %396 : tensor<1x14x14x384xf32>
      mhlo.return %397 : tensor<1x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<1x14x14xf32>, tensor<1x14x14x384xf32>, tensor<1x14x14xf32>) -> tensor<1x14x14x384xf32>
    %315 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x384x2x2xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,384,768]{1,0,2,3}"} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
      %387 = mhlo.copy %386 : tensor<2x2x384x768xf32>
      mhlo.return %387 : tensor<2x2x384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
    %316 = stablehlo.convolution(%314, %315) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x14x384xf32>, tensor<2x2x384x768xf32>) -> tensor<1x7x7x768xf32>
    %317 = "mhlo.fusion"(%316, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x7x7x768xf32>
      mhlo.return %387 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %318 = "mhlo.fusion"(%arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x768xf32>
      mhlo.return %387 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %319 = stablehlo.convolution(%317, %318) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<1x7x7x768xf32>
    %320 = "mhlo.fusion"(%319, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x7x7x768xf32>
      mhlo.return %387 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %321 = "mhlo.fusion"(%320, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
      mhlo.return %386 : tensor<1x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    %322 = "mhlo.fusion"(%321, %319, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7xf32>, %arg202: tensor<1x7x7x768xf32>, %arg203: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x7x7xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x7x7x768xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x7x7x768xf32>
      mhlo.return %393 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %323 = "mhlo.fusion"(%322, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
      mhlo.return %386 : tensor<1x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
    %324 = "mhlo.fusion"(%323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x24xf32>, tensor<f32>) -> tensor<1x7x7xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x7x7xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x7x7xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x7x7xf32>
      mhlo.return %391 : tensor<1x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x24xf32>) -> tensor<1x7x7xf32>
    %325 = "mhlo.fusion"(%arg169, %arg168, %324, %321, %319, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x7x7xf32>, %arg204: tensor<1x7x7xf32>, %arg205: tensor<1x7x7x768xf32>, %arg206: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x7x7xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x7x7x768xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x7x7x768xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x7x7x768xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x7x7x768xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x768xf32>) -> tensor<49x768xf32>
      mhlo.return %400 : tensor<49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x7x7xf32>, tensor<1x7x7xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<49x768xf32>
    %326 = stablehlo.dot_general %325, %arg170, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x768xf32>, tensor<3072x768xf32>) -> tensor<49x3072xf32>
    %327 = "mhlo.fusion"(%326, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<49x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<49x3072xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<49x3072xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %391 = stablehlo.negate %388 : tensor<49x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<49x3072xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %395 = stablehlo.abs %394 : tensor<1x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x7x7x3072xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x7x7x3072xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x7x7x3072xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x7x7x3072xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x7x7x3072xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x7x7x3072xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x7x7x3072xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x7x7x3072xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %423 = stablehlo.negate %399 : tensor<1x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %426 = stablehlo.exponential %423 : tensor<1x7x7x3072xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x7x7x3072xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x7x7x3072xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x7x7x3072xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x7x7x3072xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x7x7x3072xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x7x7x3072xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x7x7x3072xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x7x7x3072xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x7x7x3072xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x7x7x3072xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x7x7x3072xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x7x7x3072xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x7x7x3072xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x7x7x3072xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x7x7x3072xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x7x7x3072xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x7x7x3072xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x7x7x3072xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x7x7x3072xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x7x7x3072xf32>
      mhlo.return %484 : tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<49x3072xf32>, tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %328 = "mhlo.fusion"(%327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x3072xf32>, tensor<f32>) -> tensor<1x3072xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x3072xf32>
      mhlo.return %388 : tensor<1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>) -> tensor<1x3072xf32>
    %329 = mhlo.bitcast %328 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %330 = "mhlo.fusion"(%329, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x3072xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
      mhlo.return %386 : tensor<1x1x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    %331 = "mhlo.fusion"(%330, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
      mhlo.return %386 : tensor<1x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
    %332 = "mhlo.fusion"(%331, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x3xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %333 = "mhlo.fusion"(%332, %328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x3072xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x3072xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x3072xf32>
      mhlo.return %391 : tensor<1x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %334 = "mhlo.fusion"(%327, %arg173, %333, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<1x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x7x7x3072xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x7x7x3072xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x7x7x3072xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x7x7x3072xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
      mhlo.return %396 : tensor<49x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<49x3072xf32>
    %335 = stablehlo.dot_general %334, %arg174, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x3072xf32>, tensor<768x3072xf32>) -> tensor<49x768xf32>
    %336 = "mhlo.fusion"(%335, %arg175, %316, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<49x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x7x7x768xf32>, %arg204: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg203, %386 : tensor<1x7x7x768xf32>
      %388 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<49x768xf32>
      %389 = stablehlo.add %388, %arg201 : tensor<49x768xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x768xf32>) -> tensor<1x7x7x768xf32>
      %391 = stablehlo.add %387, %390 : tensor<1x7x7x768xf32>
      mhlo.return %391 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<49x768xf32>, tensor<768xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %337 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x768xf32>
      mhlo.return %387 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %338 = stablehlo.convolution(%336, %337) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<1x7x7x768xf32>
    %339 = "mhlo.fusion"(%338, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x7x7x768xf32>
      mhlo.return %387 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %340 = "mhlo.fusion"(%339, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
      mhlo.return %386 : tensor<1x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    %341 = "mhlo.fusion"(%340, %338, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7xf32>, %arg202: tensor<1x7x7x768xf32>, %arg203: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x7x7xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x7x7x768xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x7x7x768xf32>
      mhlo.return %393 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %342 = "mhlo.fusion"(%341, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
      mhlo.return %386 : tensor<1x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
    %343 = "mhlo.fusion"(%342) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x24xf32>, tensor<f32>) -> tensor<1x7x7xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x7x7xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x7x7xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x7x7xf32>
      mhlo.return %391 : tensor<1x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x24xf32>) -> tensor<1x7x7xf32>
    %344 = "mhlo.fusion"(%arg179, %arg178, %343, %340, %338, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x7x7xf32>, %arg204: tensor<1x7x7xf32>, %arg205: tensor<1x7x7x768xf32>, %arg206: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x7x7xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x7x7x768xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x7x7x768xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x7x7x768xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x7x7x768xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x768xf32>) -> tensor<49x768xf32>
      mhlo.return %400 : tensor<49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x7x7xf32>, tensor<1x7x7xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<49x768xf32>
    %345 = stablehlo.dot_general %344, %arg180, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x768xf32>, tensor<3072x768xf32>) -> tensor<49x3072xf32>
    %346 = "mhlo.fusion"(%345, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<49x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<49x3072xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<49x3072xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %391 = stablehlo.negate %388 : tensor<49x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<49x3072xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %395 = stablehlo.abs %394 : tensor<1x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x7x7x3072xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x7x7x3072xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x7x7x3072xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x7x7x3072xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x7x7x3072xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x7x7x3072xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x7x7x3072xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x7x7x3072xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %423 = stablehlo.negate %399 : tensor<1x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %426 = stablehlo.exponential %423 : tensor<1x7x7x3072xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x7x7x3072xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x7x7x3072xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x7x7x3072xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x7x7x3072xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x7x7x3072xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x7x7x3072xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x7x7x3072xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x7x7x3072xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x7x7x3072xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x7x7x3072xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x7x7x3072xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x7x7x3072xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x7x7x3072xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x7x7x3072xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x7x7x3072xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x7x7x3072xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x7x7x3072xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x7x7x3072xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x7x7x3072xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x7x7x3072xf32>
      mhlo.return %484 : tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<49x3072xf32>, tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %347 = "mhlo.fusion"(%346) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x3072xf32>, tensor<f32>) -> tensor<1x3072xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x3072xf32>
      mhlo.return %388 : tensor<1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>) -> tensor<1x3072xf32>
    %348 = mhlo.bitcast %347 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %349 = "mhlo.fusion"(%348, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x3072xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
      mhlo.return %386 : tensor<1x1x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    %350 = "mhlo.fusion"(%349, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
      mhlo.return %386 : tensor<1x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
    %351 = "mhlo.fusion"(%350, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x3xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %352 = "mhlo.fusion"(%351, %347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x3072xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x3072xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x3072xf32>
      mhlo.return %391 : tensor<1x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %353 = "mhlo.fusion"(%346, %arg183, %352, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<1x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x7x7x3072xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x7x7x3072xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x7x7x3072xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x7x7x3072xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
      mhlo.return %396 : tensor<49x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<49x3072xf32>
    %354 = stablehlo.dot_general %353, %arg184, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x3072xf32>, tensor<768x3072xf32>) -> tensor<49x768xf32>
    %355 = "mhlo.fusion"(%336, %354, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<49x768xf32>, %arg203: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<49x768xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<49x768xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x768xf32>) -> tensor<1x7x7x768xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x7x7x768xf32>
      mhlo.return %389 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<49x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %356 = "mhlo.fusion"(%arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %386 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %387 = mhlo.copy %386 : tensor<7x7x1x768xf32>
      mhlo.return %387 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %357 = stablehlo.convolution(%355, %356) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<1x7x7x768xf32>
    %358 = "mhlo.fusion"(%357, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1x7x7x768xf32>
      mhlo.return %387 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %359 = "mhlo.fusion"(%358, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
      mhlo.return %386 : tensor<1x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7xf32>
    %360 = "mhlo.fusion"(%359, %357, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7xf32>, %arg202: tensor<1x7x7x768xf32>, %arg203: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg202, %386 : tensor<1x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %389 = stablehlo.multiply %arg201, %388 : tensor<1x7x7xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x7x7x768xf32>
      %393 = stablehlo.multiply %392, %392 : tensor<1x7x7x768xf32>
      mhlo.return %393 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %361 = "mhlo.fusion"(%360, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
      mhlo.return %386 : tensor<1x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x7x7x24xf32>
    %362 = "mhlo.fusion"(%361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x7x7x24xf32>, tensor<f32>) -> tensor<1x7x7xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %387 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %388 = stablehlo.multiply %386, %387 : tensor<1x7x7xf32>
      %cst_2 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %389 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %390 = stablehlo.add %388, %389 : tensor<1x7x7xf32>
      %391 = stablehlo.rsqrt %390 : tensor<1x7x7xf32>
      mhlo.return %391 : tensor<1x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x24xf32>) -> tensor<1x7x7xf32>
    %363 = "mhlo.fusion"(%arg189, %arg188, %362, %359, %357, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x7x7xf32>, %arg204: tensor<1x7x7xf32>, %arg205: tensor<1x7x7x768xf32>, %arg206: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %387 = stablehlo.add %arg205, %386 : tensor<1x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %388 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x7xf32>
      %389 = stablehlo.multiply %arg204, %388 : tensor<1x7x7xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %391 = stablehlo.broadcast_in_dim %390, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %392 = stablehlo.subtract %387, %391 : tensor<1x7x7x768xf32>
      %393 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x7x7xf32>) -> tensor<7x7xf32>
      %394 = stablehlo.broadcast_in_dim %393, dims = [1, 2] : (tensor<7x7xf32>) -> tensor<1x7x7x768xf32>
      %395 = stablehlo.multiply %392, %394 : tensor<1x7x7x768xf32>
      %396 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %397 = stablehlo.multiply %395, %396 : tensor<1x7x7x768xf32>
      %398 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<1x7x7x768xf32>
      %399 = stablehlo.add %397, %398 : tensor<1x7x7x768xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x768xf32>) -> tensor<49x768xf32>
      mhlo.return %400 : tensor<49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x7x7xf32>, tensor<1x7x7xf32>, tensor<1x7x7x768xf32>, tensor<768xf32>) -> tensor<49x768xf32>
    %364 = stablehlo.dot_general %363, %arg190, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x768xf32>, tensor<3072x768xf32>) -> tensor<49x3072xf32>
    %365 = "mhlo.fusion"(%364, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<49x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %387 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<49x3072xf32>
      %388 = stablehlo.add %387, %arg201 : tensor<49x3072xf32>
      %389 = stablehlo.multiply %386, %388 : tensor<49x3072xf32>
      %390 = mhlo.bitcast %389 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %391 = stablehlo.negate %388 : tensor<49x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<49x3072xf32>
      %393 = stablehlo.multiply %391, %392 : tensor<49x3072xf32>
      %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x3072xf32>) -> tensor<1x7x7x3072xf32>
      %395 = stablehlo.abs %394 : tensor<1x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %396 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %397 = stablehlo.compare LT, %395, %396 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %399 = stablehlo.multiply %394, %394 : tensor<1x7x7x3072xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<1x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %402 = stablehlo.add %400, %401 : tensor<1x7x7x3072xf32>
      %403 = stablehlo.multiply %402, %399 : tensor<1x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %405 = stablehlo.add %403, %404 : tensor<1x7x7x3072xf32>
      %406 = stablehlo.multiply %405, %399 : tensor<1x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %407 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %408 = stablehlo.add %406, %407 : tensor<1x7x7x3072xf32>
      %409 = stablehlo.multiply %408, %399 : tensor<1x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %411 = stablehlo.add %409, %410 : tensor<1x7x7x3072xf32>
      %412 = stablehlo.multiply %411, %399 : tensor<1x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %414 = stablehlo.add %412, %413 : tensor<1x7x7x3072xf32>
      %415 = stablehlo.multiply %414, %399 : tensor<1x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %417 = stablehlo.add %415, %416 : tensor<1x7x7x3072xf32>
      %418 = stablehlo.multiply %394, %417 : tensor<1x7x7x3072xf32>
      %419 = stablehlo.subtract %396, %418 : tensor<1x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %421 = stablehlo.compare LT, %394, %420 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %423 = stablehlo.negate %399 : tensor<1x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %425 = stablehlo.compare LT, %423, %424 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %426 = stablehlo.exponential %423 : tensor<1x7x7x3072xf32>
      %427 = stablehlo.divide %396, %395 : tensor<1x7x7x3072xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<1x7x7x3072xf32>
      %429 = stablehlo.compare LT, %395, %422 : (tensor<1x7x7x3072xf32>, tensor<1x7x7x3072xf32>) -> tensor<1x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %431 = stablehlo.divide %396, %399 : tensor<1x7x7x3072xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<1x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %433 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %434 = stablehlo.add %432, %433 : tensor<1x7x7x3072xf32>
      %435 = stablehlo.multiply %434, %431 : tensor<1x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %437 = stablehlo.add %435, %436 : tensor<1x7x7x3072xf32>
      %438 = stablehlo.multiply %437, %431 : tensor<1x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %440 = stablehlo.add %438, %439 : tensor<1x7x7x3072xf32>
      %441 = stablehlo.multiply %440, %431 : tensor<1x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %443 = stablehlo.add %441, %442 : tensor<1x7x7x3072xf32>
      %444 = stablehlo.multiply %443, %431 : tensor<1x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %446 = stablehlo.add %444, %445 : tensor<1x7x7x3072xf32>
      %447 = stablehlo.multiply %446, %431 : tensor<1x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %449 = stablehlo.add %447, %448 : tensor<1x7x7x3072xf32>
      %450 = stablehlo.multiply %449, %431 : tensor<1x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %452 = stablehlo.add %450, %451 : tensor<1x7x7x3072xf32>
      %453 = stablehlo.multiply %452, %431 : tensor<1x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %455 = stablehlo.add %453, %454 : tensor<1x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %457 = stablehlo.multiply %456, %431 : tensor<1x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %459 = stablehlo.add %457, %458 : tensor<1x7x7x3072xf32>
      %460 = stablehlo.multiply %459, %431 : tensor<1x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %462 = stablehlo.add %460, %461 : tensor<1x7x7x3072xf32>
      %463 = stablehlo.multiply %462, %431 : tensor<1x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %465 = stablehlo.add %463, %464 : tensor<1x7x7x3072xf32>
      %466 = stablehlo.multiply %465, %431 : tensor<1x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x7x7x3072xf32>
      %469 = stablehlo.multiply %468, %431 : tensor<1x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x7x7x3072xf32>
      %472 = stablehlo.multiply %471, %431 : tensor<1x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x7x7x3072xf32>
      %475 = stablehlo.multiply %474, %431 : tensor<1x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x7x7x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x7x7x3072xf32>
      %478 = stablehlo.select %429, %455, %477 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %479 = stablehlo.multiply %428, %478 : tensor<1x7x7x3072xf32>
      %480 = stablehlo.select %425, %420, %479 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %481 = stablehlo.subtract %422, %480 : tensor<1x7x7x3072xf32>
      %482 = stablehlo.select %421, %481, %480 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %483 = stablehlo.select %397, %419, %482 : tensor<1x7x7x3072xi1>, tensor<1x7x7x3072xf32>
      %484 = stablehlo.multiply %390, %483 : tensor<1x7x7x3072xf32>
      mhlo.return %484 : tensor<1x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<49x3072xf32>, tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
    %366 = "mhlo.fusion"(%365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>):
      %386 = stablehlo.multiply %arg201, %arg201 : tensor<1x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %387 = stablehlo.reduce(%386 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x3072xf32>, tensor<f32>) -> tensor<1x3072xf32>
      %388 = stablehlo.sqrt %387 : tensor<1x3072xf32>
      mhlo.return %388 : tensor<1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>) -> tensor<1x3072xf32>
    %367 = mhlo.bitcast %366 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %368 = "mhlo.fusion"(%367, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x3072xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
      mhlo.return %386 : tensor<1x1x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x3072xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    %369 = "mhlo.fusion"(%368, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x96xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
      mhlo.return %386 : tensor<1x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x3xf32>
    %370 = "mhlo.fusion"(%369, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1x3xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
      mhlo.return %386 : tensor<1x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1x3xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %371 = "mhlo.fusion"(%370, %366) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x1x1xf32>, %arg202: tensor<1x3072xf32>):
      %386 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<1x1x1xf32>
      %387 = stablehlo.multiply %arg201, %cst_0 : tensor<1x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<1x1x1xf32>
      %388 = stablehlo.add %387, %cst_1 : tensor<1x1x1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x1x1x3072xf32>
      %391 = stablehlo.divide %386, %390 : tensor<1x1x1x3072xf32>
      mhlo.return %391 : tensor<1x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x1x1xf32>, tensor<1x3072xf32>) -> tensor<1x1x1x3072xf32>
    %372 = "mhlo.fusion"(%365, %arg193, %371, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<1x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %386 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %387 = stablehlo.broadcast_in_dim %386, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %388 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %389 = stablehlo.broadcast_in_dim %388, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %390 = stablehlo.multiply %arg201, %389 : tensor<1x7x7x3072xf32>
      %391 = stablehlo.multiply %387, %390 : tensor<1x7x7x3072xf32>
      %392 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [3] : (tensor<3072xf32>) -> tensor<1x7x7x3072xf32>
      %394 = stablehlo.add %391, %393 : tensor<1x7x7x3072xf32>
      %395 = stablehlo.add %394, %arg201 : tensor<1x7x7x3072xf32>
      %396 = mhlo.bitcast %395 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x7x7x3072xf32>) -> tensor<49x3072xf32>
      mhlo.return %396 : tensor<49x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<49x3072xf32>
    %373 = stablehlo.dot_general %372, %arg194, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<49x3072xf32>, tensor<768x3072xf32>) -> tensor<49x768xf32>
    %374 = "mhlo.fusion"(%355, %373, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<49x768xf32>, %arg203: tensor<768xf32>):
      %386 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<49x768xf32>
      %387 = stablehlo.add %386, %arg202 : tensor<49x768xf32>
      %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<49x768xf32>) -> tensor<1x7x7x768xf32>
      %389 = stablehlo.add %arg201, %388 : tensor<1x7x7x768xf32>
      mhlo.return %389 : tensor<1x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<49x768xf32>, tensor<768xf32>) -> tensor<1x7x7x768xf32>
    %375 = "mhlo.fusion"(%374, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x7x7x768xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x768xf32>
      mhlo.return %386 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x7x7x768xf32>, tensor<f32>) -> tensor<1x768xf32>
    %376 = "mhlo.fusion"(%375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %387 = stablehlo.multiply %arg201, %386 : tensor<1x768xf32>
      mhlo.return %387 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768xf32>) -> tensor<1x768xf32>
    %377 = "mhlo.fusion"(%376, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x768xf32>, tensor<f32>) -> tensor<1x24xf32>
      mhlo.return %386 : tensor<1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768xf32>, tensor<f32>) -> tensor<1x24xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x24xf32>, %arg202: tensor<f32>):
      %386 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1] : (tensor<1x24xf32>, tensor<f32>) -> tensor<1xf32>
      mhlo.return %386 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x24xf32>, tensor<f32>) -> tensor<1xf32>
    %379 = "mhlo.fusion"(%378, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1xf32>, %arg202: tensor<1x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %387 = stablehlo.multiply %arg202, %386 : tensor<1x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %388 = stablehlo.multiply %arg201, %cst_1 : tensor<1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %391 = stablehlo.subtract %387, %390 : tensor<1x768xf32>
      %392 = stablehlo.multiply %391, %391 : tensor<1x768xf32>
      mhlo.return %392 : tensor<1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1xf32>, tensor<1x768xf32>) -> tensor<1x768xf32>
    %380 = "mhlo.fusion"(%379, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x768xf32>, %arg202: tensor<f32>):
      %386 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %387 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %387 : tensor<f32>
      }) : (tensor<1x768xf32>, tensor<f32>) -> tensor<1x24xf32>
      mhlo.return %386 : tensor<1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x768xf32>, tensor<f32>) -> tensor<1x24xf32>
    %381 = "mhlo.fusion"(%380) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %386 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<1x24xf32>, tensor<f32>) -> tensor<1xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %387 = stablehlo.multiply %386, %cst_1 : tensor<1xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<1xf32>
      %388 = stablehlo.add %387, %cst_2 : tensor<1xf32>
      %389 = stablehlo.rsqrt %388 : tensor<1xf32>
      mhlo.return %389 : tensor<1xf32>
    }) {output_operand_aliasing = []} : (tensor<1x24xf32>) -> tensor<1xf32>
    %382 = "mhlo.fusion"(%arg197, %arg196, %381, %378, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1xf32>, %arg204: tensor<1xf32>, %arg205: tensor<1x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %386 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %387 = stablehlo.multiply %arg205, %386 : tensor<1x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<1xf32>
      %388 = stablehlo.multiply %arg204, %cst_1 : tensor<1xf32>
      %389 = mhlo.bitcast %388 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %390 = stablehlo.broadcast_in_dim %389, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %391 = stablehlo.subtract %387, %390 : tensor<1x768xf32>
      %392 = mhlo.bitcast %arg203 {result_layout = dense<> : tensor<0xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1xf32>) -> tensor<f32>
      %393 = stablehlo.broadcast_in_dim %392, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
      %394 = stablehlo.multiply %391, %393 : tensor<1x768xf32>
      %395 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %396 = stablehlo.multiply %394, %395 : tensor<1x768xf32>
      %397 = mhlo.bitcast %arg201 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<768xf32>) -> tensor<1x768xf32>
      %398 = stablehlo.add %396, %397 : tensor<1x768xf32>
      %399 = mhlo.bitcast %398 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x768xf32>) -> tensor<768xf32>
      mhlo.return %399 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x768xf32>) -> tensor<768xf32>
    %383 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1000x768xf32>):
      %386 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,1000]{0,1}"} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
      %387 = mhlo.copy %386 : tensor<768x1000xf32>
      mhlo.return %387 : tensor<768x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
    %384 = "mhlo.fusion"(%arg199, %382, %383) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg201: tensor<1000xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768x1000xf32>):
      %386 = stablehlo.dot %arg202, %arg203, precision = [DEFAULT, DEFAULT] : (tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
      %387 = stablehlo.add %arg201, %386 : tensor<1000xf32>
      mhlo.return %387 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
    %385 = mhlo.bitcast %384 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %385 : tensor<1x1000xf32>
  }
}
