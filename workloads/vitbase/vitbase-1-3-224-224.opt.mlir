module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_62.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_61.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_59.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_57.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_56.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_54.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_52.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_51.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_49.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_47.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_46.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_44.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_42.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_41.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_39.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_37.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_36.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_34.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_32.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_31.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_29.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_27.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_26.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_24.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.23(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_22.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.24(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_21.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.25(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_19.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_17.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_16.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.28(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_14.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.29(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_12.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_11.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.31(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_9.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_7.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_6.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.34(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @region_4.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.36(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    return %0 : tensor<1x197xf32>
  }
  func.func private @fused_computation.37(%arg0: tensor<1000xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768x1000xf32>) -> tensor<1000xf32> {
    %0 = stablehlo.dot %arg1, %arg2, precision = [DEFAULT, DEFAULT] : (tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func.func private @fused_computation.38(%arg0: tensor<1000x768xf32>) -> tensor<768x1000xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,1000]{0,1}"} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
    %1 = mhlo.copy %0 : tensor<768x1000xf32>
    return %1 : tensor<768x1000xf32>
  }
  func.func private @fused_computation.39(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = stablehlo.slice %11 [0:1, 0:1, 0:768] : (tensor<1x197x768xf32>) -> tensor<1x1x768xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
    return %13 : tensor<768xf32>
  }
  func.func private @region_2.3.clone.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.40(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.41(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.42(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.48(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.51(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.52(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.53(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.54(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.55(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.56(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.58(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.59(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.20.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.61(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.62(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.69(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.19.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.72(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.18.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.77(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.85(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.88(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.16.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.93(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.101(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.15.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.104(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.14.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.109(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.117(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.120(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.125(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.133(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.136(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.141(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.149(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.152(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.157(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.165(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.168(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.173(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.181(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.184(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.189(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.192(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.195(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.197(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.200(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.205(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.207(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.210(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.213(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.216(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.219(%arg0: tensor<197x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<197x3072xf32> {
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
    %99 = mhlo.bitcast %98 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
    return %99 : tensor<197x3072xf32>
  }
  func.func private @fused_computation.220(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.221(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<1x197x768xf32>, %arg1: tensor<197x768xf32>, %arg2: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<1x197x768xf32>
    return %3 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<12x197x64xf32>) -> tensor<197x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
    %2 = mhlo.copy %1 : tensor<1x197x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
    return %3 : tensor<197x768xf32>
  }
  func.func private @fused_computation.225(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %4 = mhlo.copy %3 : tensor<1x12x197x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %5 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<1x12x197xf32>, %arg2: tensor<1x12x197xi1>) -> tensor<12x197x197xf32> {
    %0 = stablehlo.not %arg2 : tensor<1x12x197xi1>
    %1 = mhlo.bitcast %0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %4 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %6 = stablehlo.divide %arg0, %5 : tensor<1x12x197x197xf32>
    %7 = stablehlo.select %2, %3, %6 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
    return %8 : tensor<12x197x197xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<1x12x197xf32>, %arg1: tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1x12x197xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
    %5 = stablehlo.subtract %0, %4 : tensor<1x12x197x197xf32>
    %6 = stablehlo.exponential %5 : tensor<1x12x197x197xf32>
    return %6 : tensor<1x12x197x197xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %3 = stablehlo.not %2 : tensor<1x12x197x197xi1>
    return %3 : tensor<1x12x197x197xi1>
  }
  func.func private @fused_computation.229(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x64x197xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
    %6 = mhlo.copy %5 : tensor<1x12x64x197xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
    return %7 : tensor<12x64x197xf32>
  }
  func.func private @fused_computation.230(%arg0: tensor<197x768xf32>, %arg1: tensor<768xf32>) -> tensor<12x197x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<197x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<197x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
    %6 = mhlo.copy %5 : tensor<1x12x197x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
    return %7 : tensor<12x197x64xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<1x197xf32>, %arg3: tensor<1x197x768xf32>, %arg4: tensor<1x197xf32>) -> tensor<197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg3, %3 : tensor<1x197x768xf32>
    %5 = mhlo.bitcast %arg2 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<1x197x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x197x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<1x197x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
    return %12 : tensor<197x768xf32>
  }
  func.func private @region_2.3.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.232(%arg0: tensor<1x197x24xf32>) -> tensor<1x197xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x197xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x197xf32>
    %5 = stablehlo.rsqrt %4 : tensor<1x197xf32>
    return %5 : tensor<1x197xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197xf32>) -> tensor<1x197x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<1x197xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
    %4 = stablehlo.subtract %arg0, %3 : tensor<1x197x768xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<1x197x768xf32>
    return %5 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x1x768xf32>, %arg2: tensor<1x14x14x768xf32>, %arg3: tensor<768xf32>) -> tensor<1x197x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<768xf32>) -> tensor<1x14x14x768xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<1x14x14x768xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,768,14,14]{1,3,2,0}"} : (tensor<1x14x14x768xf32>) -> tensor<1x768x14x14xf32>
    %3 = mhlo.copy %2 : tensor<1x768x14x14xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,768]{1,2,0}"} : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
    %6 = mhlo.copy %5 : tensor<1x196x768xf32>
    %7 = stablehlo.concatenate %arg1, %6, dim = 1 : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
    %8 = stablehlo.add %7, %arg0 : tensor<1x197x768xf32>
    return %8 : tensor<1x197x768xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[16,16,3,768]{1,0,2,3}"} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
    %1 = mhlo.copy %0 : tensor<16x16x3x768xf32>
    return %1 : tensor<16x16x3x768xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<1x224x224x3xf32>
    return %1 : tensor<1x224x224x3xf32>
  }
  func.func private @region_2.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_3.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_5.8(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_5.8.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_8.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_8.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_10.14(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_10.14.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_13.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_13.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_15.19(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_15.19.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_18.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_18.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_20.24(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_20.24.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_23.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_23.27.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_25.29(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_25.29.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_28.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_28.32.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_30.34(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_30.34.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_33.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_33.37.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_35.39(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_35.39.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_38.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_38.42.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_40.44(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_40.44.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.15(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_43.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.33(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_43.47.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.16(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_45.49(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.34(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_45.49.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.17(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.35(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.36(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_48.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.37(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_48.52.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.18(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_50.54(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.38(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_50.54.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.19(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.39(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.40(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_53.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.41(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_53.57.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.20(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_55.59(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.42(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_55.59.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.21(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.43(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.44(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_58.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.45(%arg0: tensor<1x12x197x197xf32>, %arg1: tensor<f32>) -> tensor<1x12x197x7xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    return %0 : tensor<1x12x197x7xf32>
  }
  func.func private @region_58.62.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.22(%arg0: tensor<1x12x197x7xf32>, %arg1: tensor<f32>) -> tensor<1x12x197xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    return %0 : tensor<1x12x197xf32>
  }
  func.func private @region_60.64(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.46(%arg0: tensor<1x12x197x197xi1>, %arg1: tensor<i1>) -> tensor<1x12x197x7xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    return %0 : tensor<1x12x197x7xi1>
  }
  func.func private @region_60.64.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.23(%arg0: tensor<1x12x197x7xi1>, %arg1: tensor<i1>) -> tensor<1x12x197xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    return %0 : tensor<1x12x197xi1>
  }
  func.func private @region_2.3.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.47(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func private @region_2.3.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.48(%arg0: tensor<1x197x768xf32>, %arg1: tensor<f32>) -> tensor<1x197x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    return %0 : tensor<1x197x24xf32>
  }
  func.func @main(%arg0: tensor<1x1x768xf32>, %arg1: tensor<1x197x768xf32>, %arg2: tensor<768x3x16x16xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768x768xf32>, %arg7: tensor<768xf32>, %arg8: tensor<768x768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768x768xf32>, %arg11: tensor<768xf32>, %arg12: tensor<3072x768xf32>, %arg13: tensor<3072xf32>, %arg14: tensor<768x3072xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768x768xf32>, %arg21: tensor<768xf32>, %arg22: tensor<768x768xf32>, %arg23: tensor<768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768xf32>, %arg26: tensor<768x768xf32>, %arg27: tensor<768xf32>, %arg28: tensor<3072x768xf32>, %arg29: tensor<3072xf32>, %arg30: tensor<768x3072xf32>, %arg31: tensor<768xf32>, %arg32: tensor<768xf32>, %arg33: tensor<768xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768x768xf32>, %arg37: tensor<768xf32>, %arg38: tensor<768x768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768x768xf32>, %arg41: tensor<768xf32>, %arg42: tensor<768x768xf32>, %arg43: tensor<768xf32>, %arg44: tensor<3072x768xf32>, %arg45: tensor<3072xf32>, %arg46: tensor<768x3072xf32>, %arg47: tensor<768xf32>, %arg48: tensor<768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768x768xf32>, %arg53: tensor<768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768x768xf32>, %arg57: tensor<768xf32>, %arg58: tensor<768x768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<3072x768xf32>, %arg61: tensor<3072xf32>, %arg62: tensor<768x3072xf32>, %arg63: tensor<768xf32>, %arg64: tensor<768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768x768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768x768xf32>, %arg71: tensor<768xf32>, %arg72: tensor<768x768xf32>, %arg73: tensor<768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768xf32>, %arg76: tensor<3072x768xf32>, %arg77: tensor<3072xf32>, %arg78: tensor<768x3072xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768x768xf32>, %arg87: tensor<768xf32>, %arg88: tensor<768x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768x768xf32>, %arg91: tensor<768xf32>, %arg92: tensor<3072x768xf32>, %arg93: tensor<3072xf32>, %arg94: tensor<768x3072xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768x768xf32>, %arg101: tensor<768xf32>, %arg102: tensor<768x768xf32>, %arg103: tensor<768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<768x768xf32>, %arg107: tensor<768xf32>, %arg108: tensor<3072x768xf32>, %arg109: tensor<3072xf32>, %arg110: tensor<768x3072xf32>, %arg111: tensor<768xf32>, %arg112: tensor<768xf32>, %arg113: tensor<768xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768x768xf32>, %arg117: tensor<768xf32>, %arg118: tensor<768x768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768x768xf32>, %arg121: tensor<768xf32>, %arg122: tensor<768x768xf32>, %arg123: tensor<768xf32>, %arg124: tensor<3072x768xf32>, %arg125: tensor<3072xf32>, %arg126: tensor<768x3072xf32>, %arg127: tensor<768xf32>, %arg128: tensor<768xf32>, %arg129: tensor<768xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768x768xf32>, %arg133: tensor<768xf32>, %arg134: tensor<768x768xf32>, %arg135: tensor<768xf32>, %arg136: tensor<768x768xf32>, %arg137: tensor<768xf32>, %arg138: tensor<768x768xf32>, %arg139: tensor<768xf32>, %arg140: tensor<3072x768xf32>, %arg141: tensor<3072xf32>, %arg142: tensor<768x3072xf32>, %arg143: tensor<768xf32>, %arg144: tensor<768xf32>, %arg145: tensor<768xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768x768xf32>, %arg149: tensor<768xf32>, %arg150: tensor<768x768xf32>, %arg151: tensor<768xf32>, %arg152: tensor<768x768xf32>, %arg153: tensor<768xf32>, %arg154: tensor<768x768xf32>, %arg155: tensor<768xf32>, %arg156: tensor<3072x768xf32>, %arg157: tensor<3072xf32>, %arg158: tensor<768x3072xf32>, %arg159: tensor<768xf32>, %arg160: tensor<768xf32>, %arg161: tensor<768xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768x768xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x768xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768x768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<768x768xf32>, %arg171: tensor<768xf32>, %arg172: tensor<3072x768xf32>, %arg173: tensor<3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768x768xf32>, %arg181: tensor<768xf32>, %arg182: tensor<768x768xf32>, %arg183: tensor<768xf32>, %arg184: tensor<768x768xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x768xf32>, %arg187: tensor<768xf32>, %arg188: tensor<3072x768xf32>, %arg189: tensor<3072xf32>, %arg190: tensor<768x3072xf32>, %arg191: tensor<768xf32>, %arg192: tensor<768xf32>, %arg193: tensor<768xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x3x224x224xf32>):
      %420 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,224,224,3]{2,1,3,0}"} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
      %421 = mhlo.copy %420 : tensor<1x224x224x3xf32>
      mhlo.return %421 : tensor<1x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg2) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x16x16xf32>):
      %420 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[16,16,3,768]{1,0,2,3}"} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
      %421 = mhlo.copy %420 : tensor<16x16x3x768xf32>
      mhlo.return %421 : tensor<16x16x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x16x16xf32>) -> tensor<16x16x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [16, 16], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x224x224x3xf32>, tensor<16x16x3x768xf32>) -> tensor<1x14x14x768xf32>
    %3 = "mhlo.fusion"(%arg1, %arg0, %2, %arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x1x768xf32>, %arg203: tensor<1x14x14x768xf32>, %arg204: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<768xf32>) -> tensor<1x14x14x768xf32>
      %421 = stablehlo.add %arg203, %420 : tensor<1x14x14x768xf32>
      %422 = stablehlo.transpose %421, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,768,14,14]{1,3,2,0}"} : (tensor<1x14x14x768xf32>) -> tensor<1x768x14x14xf32>
      %423 = mhlo.copy %422 : tensor<1x768x14x14xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,196,768]{1,2,0}"} : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
      %426 = mhlo.copy %425 : tensor<1x196x768xf32>
      %427 = stablehlo.concatenate %arg202, %426, dim = 1 : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
      %428 = stablehlo.add %427, %arg201 : tensor<1x197x768xf32>
      mhlo.return %428 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x1x768xf32>, tensor<1x14x14x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.fusion"(%3, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %5 = "mhlo.fusion"(%3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %7 = "mhlo.fusion"(%6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %8 = "mhlo.fusion"(%arg17, %arg16, %7, %3, %4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %9 = stablehlo.dot_general %8, %arg4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %10 = "mhlo.fusion"(%9, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %11 = stablehlo.dot_general %8, %arg6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %12 = "mhlo.fusion"(%11, %arg7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %13 = stablehlo.dot_general %10, %12, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = "mhlo.fusion"(%14, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %16 = "mhlo.fusion"(%15, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %17 = "mhlo.fusion"(%16, %13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %18 = "mhlo.fusion"(%17, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %19 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %20 = "mhlo.fusion"(%19, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %21 = "mhlo.fusion"(%20, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %22 = "mhlo.fusion"(%17, %18, %21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %23 = stablehlo.dot_general %8, %arg8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %24 = "mhlo.fusion"(%23, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %25 = stablehlo.dot_general %22, %24, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %26 = "mhlo.fusion"(%25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %27 = stablehlo.dot_general %26, %arg10, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %28 = "mhlo.fusion"(%3, %27, %arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %29 = "mhlo.fusion"(%28, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %30 = "mhlo.fusion"(%28, %29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %31 = "mhlo.fusion"(%30, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %32 = "mhlo.fusion"(%31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %33 = "mhlo.fusion"(%arg19, %arg18, %32, %28, %29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %34 = stablehlo.dot_general %33, %arg12, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %35 = "mhlo.fusion"(%34, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %36 = stablehlo.dot_general %35, %arg14, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %37 = "mhlo.fusion"(%28, %36, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %38 = "mhlo.fusion"(%37, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %39 = "mhlo.fusion"(%37, %38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %40 = "mhlo.fusion"(%39, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %41 = "mhlo.fusion"(%40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %42 = "mhlo.fusion"(%arg33, %arg32, %41, %37, %38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %43 = stablehlo.dot_general %42, %arg20, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %44 = "mhlo.fusion"(%43, %arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %45 = stablehlo.dot_general %42, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %46 = "mhlo.fusion"(%45, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %47 = stablehlo.dot_general %44, %46, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %49 = "mhlo.fusion"(%48, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %50 = "mhlo.fusion"(%49, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %51 = "mhlo.fusion"(%50, %47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %52 = "mhlo.fusion"(%51, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %53 = "mhlo.fusion"(%47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %54 = "mhlo.fusion"(%53, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %55 = "mhlo.fusion"(%54, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %56 = "mhlo.fusion"(%51, %52, %55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %57 = stablehlo.dot_general %42, %arg24, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %58 = "mhlo.fusion"(%57, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %59 = stablehlo.dot_general %56, %58, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %60 = "mhlo.fusion"(%59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %61 = stablehlo.dot_general %60, %arg26, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %62 = "mhlo.fusion"(%37, %61, %arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %63 = "mhlo.fusion"(%62, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %64 = "mhlo.fusion"(%62, %63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %65 = "mhlo.fusion"(%64, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %66 = "mhlo.fusion"(%65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %67 = "mhlo.fusion"(%arg35, %arg34, %66, %62, %63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %68 = stablehlo.dot_general %67, %arg28, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %69 = "mhlo.fusion"(%68, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %70 = stablehlo.dot_general %69, %arg30, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %71 = "mhlo.fusion"(%62, %70, %arg31) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %72 = "mhlo.fusion"(%71, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %74 = "mhlo.fusion"(%73, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %75 = "mhlo.fusion"(%74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %76 = "mhlo.fusion"(%arg49, %arg48, %75, %71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %77 = stablehlo.dot_general %76, %arg36, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %78 = "mhlo.fusion"(%77, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %79 = stablehlo.dot_general %76, %arg38, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %80 = "mhlo.fusion"(%79, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %81 = stablehlo.dot_general %78, %80, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %83 = "mhlo.fusion"(%82, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %84 = "mhlo.fusion"(%83, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %85 = "mhlo.fusion"(%84, %81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %86 = "mhlo.fusion"(%85, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %87 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %88 = "mhlo.fusion"(%87, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %89 = "mhlo.fusion"(%88, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %90 = "mhlo.fusion"(%85, %86, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %91 = stablehlo.dot_general %76, %arg40, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %92 = "mhlo.fusion"(%91, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %93 = stablehlo.dot_general %90, %92, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %95 = stablehlo.dot_general %94, %arg42, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %96 = "mhlo.fusion"(%71, %95, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %97 = "mhlo.fusion"(%96, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %98 = "mhlo.fusion"(%96, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %99 = "mhlo.fusion"(%98, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %100 = "mhlo.fusion"(%99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %101 = "mhlo.fusion"(%arg51, %arg50, %100, %96, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %102 = stablehlo.dot_general %101, %arg44, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %103 = "mhlo.fusion"(%102, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %104 = stablehlo.dot_general %103, %arg46, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %105 = "mhlo.fusion"(%96, %104, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %107 = "mhlo.fusion"(%105, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %108 = "mhlo.fusion"(%107, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %110 = "mhlo.fusion"(%arg65, %arg64, %109, %105, %106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %111 = stablehlo.dot_general %110, %arg52, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %112 = "mhlo.fusion"(%111, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %113 = stablehlo.dot_general %110, %arg54, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %114 = "mhlo.fusion"(%113, %arg55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %115 = stablehlo.dot_general %112, %114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %117 = "mhlo.fusion"(%116, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %118 = "mhlo.fusion"(%117, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %119 = "mhlo.fusion"(%118, %115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %120 = "mhlo.fusion"(%119, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %121 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %122 = "mhlo.fusion"(%121, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %123 = "mhlo.fusion"(%122, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %124 = "mhlo.fusion"(%119, %120, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %125 = stablehlo.dot_general %110, %arg56, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %126 = "mhlo.fusion"(%125, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %127 = stablehlo.dot_general %124, %126, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %128 = "mhlo.fusion"(%127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %129 = stablehlo.dot_general %128, %arg58, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %130 = "mhlo.fusion"(%105, %129, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %131 = "mhlo.fusion"(%130, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %132 = "mhlo.fusion"(%130, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %133 = "mhlo.fusion"(%132, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %135 = "mhlo.fusion"(%arg67, %arg66, %134, %130, %131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %136 = stablehlo.dot_general %135, %arg60, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %137 = "mhlo.fusion"(%136, %arg61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %138 = stablehlo.dot_general %137, %arg62, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %139 = "mhlo.fusion"(%130, %138, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %140 = "mhlo.fusion"(%139, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %141 = "mhlo.fusion"(%139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %144 = "mhlo.fusion"(%arg81, %arg80, %143, %139, %140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %145 = stablehlo.dot_general %144, %arg68, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %146 = "mhlo.fusion"(%145, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %147 = stablehlo.dot_general %144, %arg70, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %148 = "mhlo.fusion"(%147, %arg71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %149 = stablehlo.dot_general %146, %148, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %150 = mhlo.bitcast %149 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %151 = "mhlo.fusion"(%150, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %152 = "mhlo.fusion"(%151, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %153 = "mhlo.fusion"(%152, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %154 = "mhlo.fusion"(%153, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %155 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %156 = "mhlo.fusion"(%155, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %157 = "mhlo.fusion"(%156, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %158 = "mhlo.fusion"(%153, %154, %157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %159 = stablehlo.dot_general %144, %arg72, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %160 = "mhlo.fusion"(%159, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %161 = stablehlo.dot_general %158, %160, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %162 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %163 = stablehlo.dot_general %162, %arg74, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %164 = "mhlo.fusion"(%139, %163, %arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %165 = "mhlo.fusion"(%164, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %166 = "mhlo.fusion"(%164, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %167 = "mhlo.fusion"(%166, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %168 = "mhlo.fusion"(%167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %169 = "mhlo.fusion"(%arg83, %arg82, %168, %164, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %170 = stablehlo.dot_general %169, %arg76, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %171 = "mhlo.fusion"(%170, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %172 = stablehlo.dot_general %171, %arg78, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %173 = "mhlo.fusion"(%164, %172, %arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %174 = "mhlo.fusion"(%173, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %175 = "mhlo.fusion"(%173, %174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %176 = "mhlo.fusion"(%175, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %178 = "mhlo.fusion"(%arg97, %arg96, %177, %173, %174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %179 = stablehlo.dot_general %178, %arg84, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %180 = "mhlo.fusion"(%179, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %181 = stablehlo.dot_general %178, %arg86, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %182 = "mhlo.fusion"(%181, %arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %183 = stablehlo.dot_general %180, %182, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %184 = mhlo.bitcast %183 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %185 = "mhlo.fusion"(%184, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %186 = "mhlo.fusion"(%185, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %187 = "mhlo.fusion"(%186, %183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %188 = "mhlo.fusion"(%187, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %189 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %190 = "mhlo.fusion"(%189, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %191 = "mhlo.fusion"(%190, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %192 = "mhlo.fusion"(%187, %188, %191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %193 = stablehlo.dot_general %178, %arg88, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %194 = "mhlo.fusion"(%193, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %195 = stablehlo.dot_general %192, %194, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %197 = stablehlo.dot_general %196, %arg90, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %198 = "mhlo.fusion"(%173, %197, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %200 = "mhlo.fusion"(%198, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %201 = "mhlo.fusion"(%200, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %202 = "mhlo.fusion"(%201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %203 = "mhlo.fusion"(%arg99, %arg98, %202, %198, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %204 = stablehlo.dot_general %203, %arg92, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %205 = "mhlo.fusion"(%204, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %206 = stablehlo.dot_general %205, %arg94, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %207 = "mhlo.fusion"(%198, %206, %arg95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %208 = "mhlo.fusion"(%207, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %209 = "mhlo.fusion"(%207, %208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %210 = "mhlo.fusion"(%209, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %212 = "mhlo.fusion"(%arg113, %arg112, %211, %207, %208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %213 = stablehlo.dot_general %212, %arg100, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %214 = "mhlo.fusion"(%213, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %215 = stablehlo.dot_general %212, %arg102, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %216 = "mhlo.fusion"(%215, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %217 = stablehlo.dot_general %214, %216, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %219 = "mhlo.fusion"(%218, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %220 = "mhlo.fusion"(%219, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %221 = "mhlo.fusion"(%220, %217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %222 = "mhlo.fusion"(%221, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %223 = "mhlo.fusion"(%217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %224 = "mhlo.fusion"(%223, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %225 = "mhlo.fusion"(%224, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %226 = "mhlo.fusion"(%221, %222, %225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %227 = stablehlo.dot_general %212, %arg104, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %228 = "mhlo.fusion"(%227, %arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %229 = stablehlo.dot_general %226, %228, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %230 = "mhlo.fusion"(%229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %231 = stablehlo.dot_general %230, %arg106, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %232 = "mhlo.fusion"(%207, %231, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %233 = "mhlo.fusion"(%232, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %234 = "mhlo.fusion"(%232, %233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %235 = "mhlo.fusion"(%234, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %236 = "mhlo.fusion"(%235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %237 = "mhlo.fusion"(%arg115, %arg114, %236, %232, %233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %238 = stablehlo.dot_general %237, %arg108, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %239 = "mhlo.fusion"(%238, %arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %240 = stablehlo.dot_general %239, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %241 = "mhlo.fusion"(%232, %240, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %242 = "mhlo.fusion"(%241, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %243 = "mhlo.fusion"(%241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %244 = "mhlo.fusion"(%243, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %245 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %246 = "mhlo.fusion"(%arg129, %arg128, %245, %241, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %247 = stablehlo.dot_general %246, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %248 = "mhlo.fusion"(%247, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %249 = stablehlo.dot_general %246, %arg118, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %250 = "mhlo.fusion"(%249, %arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %251 = stablehlo.dot_general %248, %250, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %253 = "mhlo.fusion"(%252, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %254 = "mhlo.fusion"(%253, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %255 = "mhlo.fusion"(%254, %251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %257 = "mhlo.fusion"(%251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %258 = "mhlo.fusion"(%257, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %259 = "mhlo.fusion"(%258, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %260 = "mhlo.fusion"(%255, %256, %259) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %261 = stablehlo.dot_general %246, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %262 = "mhlo.fusion"(%261, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %263 = stablehlo.dot_general %260, %262, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %264 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %265 = stablehlo.dot_general %264, %arg122, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %266 = "mhlo.fusion"(%241, %265, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %267 = "mhlo.fusion"(%266, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %268 = "mhlo.fusion"(%266, %267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %269 = "mhlo.fusion"(%268, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %270 = "mhlo.fusion"(%269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %271 = "mhlo.fusion"(%arg131, %arg130, %270, %266, %267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %272 = stablehlo.dot_general %271, %arg124, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %273 = "mhlo.fusion"(%272, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %274 = stablehlo.dot_general %273, %arg126, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %275 = "mhlo.fusion"(%266, %274, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %276 = "mhlo.fusion"(%275, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %277 = "mhlo.fusion"(%275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %278 = "mhlo.fusion"(%277, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %279 = "mhlo.fusion"(%278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %280 = "mhlo.fusion"(%arg145, %arg144, %279, %275, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %281 = stablehlo.dot_general %280, %arg132, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %282 = "mhlo.fusion"(%281, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %283 = stablehlo.dot_general %280, %arg134, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %284 = "mhlo.fusion"(%283, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %285 = stablehlo.dot_general %282, %284, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %287 = "mhlo.fusion"(%286, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %288 = "mhlo.fusion"(%287, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %289 = "mhlo.fusion"(%288, %285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %290 = "mhlo.fusion"(%289, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %291 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %292 = "mhlo.fusion"(%291, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %293 = "mhlo.fusion"(%292, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %294 = "mhlo.fusion"(%289, %290, %293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %295 = stablehlo.dot_general %280, %arg136, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %296 = "mhlo.fusion"(%295, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %297 = stablehlo.dot_general %294, %296, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %298 = "mhlo.fusion"(%297) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %299 = stablehlo.dot_general %298, %arg138, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %300 = "mhlo.fusion"(%275, %299, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %301 = "mhlo.fusion"(%300, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %302 = "mhlo.fusion"(%300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %303 = "mhlo.fusion"(%302, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %304 = "mhlo.fusion"(%303) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %305 = "mhlo.fusion"(%arg147, %arg146, %304, %300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %306 = stablehlo.dot_general %305, %arg140, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %307 = "mhlo.fusion"(%306, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %308 = stablehlo.dot_general %307, %arg142, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %309 = "mhlo.fusion"(%300, %308, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %311 = "mhlo.fusion"(%309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %312 = "mhlo.fusion"(%311, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %313 = "mhlo.fusion"(%312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %314 = "mhlo.fusion"(%arg161, %arg160, %313, %309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %315 = stablehlo.dot_general %314, %arg148, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %316 = "mhlo.fusion"(%315, %arg149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %317 = stablehlo.dot_general %314, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %318 = "mhlo.fusion"(%317, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %319 = stablehlo.dot_general %316, %318, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %320 = mhlo.bitcast %319 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %322 = "mhlo.fusion"(%321, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %323 = "mhlo.fusion"(%322, %319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %324 = "mhlo.fusion"(%323, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %325 = "mhlo.fusion"(%319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %326 = "mhlo.fusion"(%325, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %327 = "mhlo.fusion"(%326, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %328 = "mhlo.fusion"(%323, %324, %327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %329 = stablehlo.dot_general %314, %arg152, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %330 = "mhlo.fusion"(%329, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %331 = stablehlo.dot_general %328, %330, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %332 = "mhlo.fusion"(%331) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %333 = stablehlo.dot_general %332, %arg154, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %334 = "mhlo.fusion"(%309, %333, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %335 = "mhlo.fusion"(%334, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %336 = "mhlo.fusion"(%334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %338 = "mhlo.fusion"(%337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %339 = "mhlo.fusion"(%arg163, %arg162, %338, %334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %340 = stablehlo.dot_general %339, %arg156, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %341 = "mhlo.fusion"(%340, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %342 = stablehlo.dot_general %341, %arg158, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %343 = "mhlo.fusion"(%334, %342, %arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %344 = "mhlo.fusion"(%343, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %345 = "mhlo.fusion"(%343, %344) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %346 = "mhlo.fusion"(%345, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %347 = "mhlo.fusion"(%346) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %348 = "mhlo.fusion"(%arg177, %arg176, %347, %343, %344) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %349 = stablehlo.dot_general %348, %arg164, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %350 = "mhlo.fusion"(%349, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %351 = stablehlo.dot_general %348, %arg166, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %352 = "mhlo.fusion"(%351, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %353 = stablehlo.dot_general %350, %352, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %354 = mhlo.bitcast %353 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %355 = "mhlo.fusion"(%354, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %356 = "mhlo.fusion"(%355, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %357 = "mhlo.fusion"(%356, %353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %358 = "mhlo.fusion"(%357, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %359 = "mhlo.fusion"(%353) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %360 = "mhlo.fusion"(%359, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %361 = "mhlo.fusion"(%360, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %362 = "mhlo.fusion"(%357, %358, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %363 = stablehlo.dot_general %348, %arg168, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %364 = "mhlo.fusion"(%363, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %365 = stablehlo.dot_general %362, %364, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %366 = "mhlo.fusion"(%365) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %367 = stablehlo.dot_general %366, %arg170, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %368 = "mhlo.fusion"(%343, %367, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %369 = "mhlo.fusion"(%368, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %370 = "mhlo.fusion"(%368, %369) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %371 = "mhlo.fusion"(%370, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %372 = "mhlo.fusion"(%371) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %373 = "mhlo.fusion"(%arg179, %arg178, %372, %368, %369) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %374 = stablehlo.dot_general %373, %arg172, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %375 = "mhlo.fusion"(%374, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %376 = stablehlo.dot_general %375, %arg174, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %377 = "mhlo.fusion"(%368, %376, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %379 = "mhlo.fusion"(%377, %378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %380 = "mhlo.fusion"(%379, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %381 = "mhlo.fusion"(%380) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %382 = "mhlo.fusion"(%arg193, %arg192, %381, %377, %378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %383 = stablehlo.dot_general %382, %arg180, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %384 = "mhlo.fusion"(%383, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %426 = mhlo.copy %425 : tensor<1x12x197x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %427 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %385 = stablehlo.dot_general %382, %arg182, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %386 = "mhlo.fusion"(%385, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x768xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<197x768xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,64,197]{2,1,3,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x64x197xf32>
      %426 = mhlo.copy %425 : tensor<1x12x64x197xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x64x197xf32>) -> tensor<12x64x197xf32>
      mhlo.return %427 : tensor<12x64x197xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x64x197xf32>
    %387 = stablehlo.dot_general %384, %386, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x64xf32>, tensor<12x64x197xf32>) -> tensor<12x197x197xf32>
    %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %389 = "mhlo.fusion"(%388, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
      mhlo.return %420 : tensor<1x12x197x7xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197x7xf32>
    %390 = "mhlo.fusion"(%389, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %391 = "mhlo.fusion"(%390, %387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197xf32>, %arg202: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197xf32>
      %422 = stablehlo.maximum %421, %arg201 : tensor<1x12x197xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<1x12x197x197xf32>
      %426 = stablehlo.exponential %425 : tensor<1x12x197x197xf32>
      mhlo.return %426 : tensor<1x12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197xf32>, tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
    %392 = "mhlo.fusion"(%391, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
      mhlo.return %420 : tensor<1x12x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<f32>) -> tensor<1x12x197xf32>
    %393 = "mhlo.fusion"(%387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x197xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %422 = stablehlo.compare EQ, %420, %421 : (tensor<1x12x197x197xf32>, tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xi1>
      %423 = stablehlo.not %422 : tensor<1x12x197x197xi1>
      mhlo.return %423 : tensor<1x12x197x197xi1>
    }) {output_operand_aliasing = []} : (tensor<12x197x197xf32>) -> tensor<1x12x197x197xi1>
    %394 = "mhlo.fusion"(%393, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xi1>, %arg202: tensor<i1>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [13, 14]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %421 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %421 : tensor<i1>
      }) : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
      mhlo.return %420 : tensor<1x12x197x7xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xi1>, tensor<i1>) -> tensor<1x12x197x7xi1>
    %395 = "mhlo.fusion"(%394, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x7xi1>, %arg202: tensor<i1>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
      mhlo.return %420 : tensor<1x12x197xi1>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x7xi1>, tensor<i1>) -> tensor<1x12x197xi1>
    %396 = "mhlo.fusion"(%391, %392, %395) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x12x197x197xf32>, %arg202: tensor<1x12x197xf32>, %arg203: tensor<1x12x197xi1>):
      %420 = stablehlo.not %arg203 : tensor<1x12x197xi1>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xi1>) -> tensor<12x197xi1>
      %422 = stablehlo.broadcast_in_dim %421, dims = [1, 2] : (tensor<12x197xi1>) -> tensor<1x12x197x197xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x12x197x197xf32>
      %424 = mhlo.bitcast %arg202 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x12x197xf32>) -> tensor<12x197xf32>
      %425 = stablehlo.broadcast_in_dim %424, dims = [1, 2] : (tensor<12x197xf32>) -> tensor<1x12x197x197xf32>
      %426 = stablehlo.divide %arg201, %425 : tensor<1x12x197x197xf32>
      %427 = stablehlo.select %422, %423, %426 : tensor<1x12x197x197xi1>, tensor<1x12x197x197xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x197xf32>) -> tensor<12x197x197xf32>
      mhlo.return %428 : tensor<12x197x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x12x197x197xf32>, tensor<1x12x197xf32>, tensor<1x12x197xi1>) -> tensor<12x197x197xf32>
    %397 = stablehlo.dot_general %382, %arg184, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %398 = "mhlo.fusion"(%397, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x768xf32>, %arg202: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg201 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x12x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,12,197,64]{3,1,2,0}"} : (tensor<1x197x12x64xf32>) -> tensor<1x12x197x64xf32>
      %424 = mhlo.copy %423 : tensor<1x12x197x64xf32>
      %425 = mhlo.bitcast %424 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x12x197x64xf32>) -> tensor<12x197x64xf32>
      mhlo.return %425 : tensor<12x197x64xf32>
    }) {output_operand_aliasing = []} : (tensor<197x768xf32>, tensor<768xf32>) -> tensor<12x197x64xf32>
    %399 = stablehlo.dot_general %396, %398, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<12x197x197xf32>, tensor<12x197x64xf32>) -> tensor<12x197x64xf32>
    %400 = "mhlo.fusion"(%399) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<12x197x64xf32>):
      %420 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<12x197x64xf32>) -> tensor<1x12x197x64xf32>
      %421 = stablehlo.transpose %420, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,197,12,64]{3,1,2,0}"} : (tensor<1x12x197x64xf32>) -> tensor<1x197x12x64xf32>
      %422 = mhlo.copy %421 : tensor<1x197x12x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x197x12x64xf32>) -> tensor<197x768xf32>
      mhlo.return %423 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<12x197x64xf32>) -> tensor<197x768xf32>
    %401 = stablehlo.dot_general %400, %arg186, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<768x768xf32>) -> tensor<197x768xf32>
    %402 = "mhlo.fusion"(%377, %401, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %403 = "mhlo.fusion"(%402, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %404 = "mhlo.fusion"(%402, %403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %405 = "mhlo.fusion"(%404, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %406 = "mhlo.fusion"(%405) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %407 = "mhlo.fusion"(%arg195, %arg194, %406, %402, %403) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = mhlo.bitcast %431 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x768xf32>) -> tensor<197x768xf32>
      mhlo.return %432 : tensor<197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<197x768xf32>
    %408 = stablehlo.dot_general %407, %arg188, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x768xf32>, tensor<3072x768xf32>) -> tensor<197x3072xf32>
    %409 = "mhlo.fusion"(%408, %arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<197x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %421 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<197x3072xf32>
      %422 = stablehlo.add %421, %arg201 : tensor<197x3072xf32>
      %423 = stablehlo.multiply %420, %422 : tensor<197x3072xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %425 = stablehlo.negate %422 : tensor<197x3072xf32>
      %cst_2 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %426 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<197x3072xf32>
      %427 = stablehlo.multiply %425, %426 : tensor<197x3072xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x3072xf32>) -> tensor<1x197x3072xf32>
      %429 = stablehlo.abs %428 : tensor<1x197x3072xf32>
      %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %430 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %431 = stablehlo.compare LT, %429, %430 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_4 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %433 = stablehlo.multiply %428, %428 : tensor<1x197x3072xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<1x197x3072xf32>
      %cst_5 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %436 = stablehlo.add %434, %435 : tensor<1x197x3072xf32>
      %437 = stablehlo.multiply %436, %433 : tensor<1x197x3072xf32>
      %cst_6 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %439 = stablehlo.add %437, %438 : tensor<1x197x3072xf32>
      %440 = stablehlo.multiply %439, %433 : tensor<1x197x3072xf32>
      %cst_7 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %441 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %442 = stablehlo.add %440, %441 : tensor<1x197x3072xf32>
      %443 = stablehlo.multiply %442, %433 : tensor<1x197x3072xf32>
      %cst_8 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %444 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %445 = stablehlo.add %443, %444 : tensor<1x197x3072xf32>
      %446 = stablehlo.multiply %445, %433 : tensor<1x197x3072xf32>
      %cst_9 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %448 = stablehlo.add %446, %447 : tensor<1x197x3072xf32>
      %449 = stablehlo.multiply %448, %433 : tensor<1x197x3072xf32>
      %cst_10 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %450 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %451 = stablehlo.add %449, %450 : tensor<1x197x3072xf32>
      %452 = stablehlo.multiply %428, %451 : tensor<1x197x3072xf32>
      %453 = stablehlo.subtract %430, %452 : tensor<1x197x3072xf32>
      %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %455 = stablehlo.compare LT, %428, %454 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %456 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %457 = stablehlo.negate %433 : tensor<1x197x3072xf32>
      %cst_13 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %458 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %459 = stablehlo.compare LT, %457, %458 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %460 = stablehlo.exponential %457 : tensor<1x197x3072xf32>
      %461 = stablehlo.divide %430, %429 : tensor<1x197x3072xf32>
      %462 = stablehlo.multiply %460, %461 : tensor<1x197x3072xf32>
      %463 = stablehlo.compare LT, %429, %456 : (tensor<1x197x3072xf32>, tensor<1x197x3072xf32>) -> tensor<1x197x3072xi1>
      %cst_14 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %465 = stablehlo.divide %430, %433 : tensor<1x197x3072xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<1x197x3072xf32>
      %cst_15 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %467 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %468 = stablehlo.add %466, %467 : tensor<1x197x3072xf32>
      %469 = stablehlo.multiply %468, %465 : tensor<1x197x3072xf32>
      %cst_16 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<1x197x3072xf32>
      %472 = stablehlo.multiply %471, %465 : tensor<1x197x3072xf32>
      %cst_17 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<1x197x3072xf32>
      %475 = stablehlo.multiply %474, %465 : tensor<1x197x3072xf32>
      %cst_18 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<1x197x3072xf32>
      %478 = stablehlo.multiply %477, %465 : tensor<1x197x3072xf32>
      %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<1x197x3072xf32>
      %481 = stablehlo.multiply %480, %465 : tensor<1x197x3072xf32>
      %cst_20 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<1x197x3072xf32>
      %484 = stablehlo.multiply %483, %465 : tensor<1x197x3072xf32>
      %cst_21 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<1x197x3072xf32>
      %487 = stablehlo.multiply %486, %465 : tensor<1x197x3072xf32>
      %cst_22 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<1x197x3072xf32>
      %cst_23 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %491 = stablehlo.multiply %490, %465 : tensor<1x197x3072xf32>
      %cst_24 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %492 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %493 = stablehlo.add %491, %492 : tensor<1x197x3072xf32>
      %494 = stablehlo.multiply %493, %465 : tensor<1x197x3072xf32>
      %cst_25 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %496 = stablehlo.add %494, %495 : tensor<1x197x3072xf32>
      %497 = stablehlo.multiply %496, %465 : tensor<1x197x3072xf32>
      %cst_26 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %498 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %499 = stablehlo.add %497, %498 : tensor<1x197x3072xf32>
      %500 = stablehlo.multiply %499, %465 : tensor<1x197x3072xf32>
      %cst_27 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %501 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %502 = stablehlo.add %500, %501 : tensor<1x197x3072xf32>
      %503 = stablehlo.multiply %502, %465 : tensor<1x197x3072xf32>
      %cst_28 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %504 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %505 = stablehlo.add %503, %504 : tensor<1x197x3072xf32>
      %506 = stablehlo.multiply %505, %465 : tensor<1x197x3072xf32>
      %cst_29 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %507 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %508 = stablehlo.add %506, %507 : tensor<1x197x3072xf32>
      %509 = stablehlo.multiply %508, %465 : tensor<1x197x3072xf32>
      %cst_30 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x197x3072xf32>
      %511 = stablehlo.add %509, %510 : tensor<1x197x3072xf32>
      %512 = stablehlo.select %463, %489, %511 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %513 = stablehlo.multiply %462, %512 : tensor<1x197x3072xf32>
      %514 = stablehlo.select %459, %454, %513 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %515 = stablehlo.subtract %456, %514 : tensor<1x197x3072xf32>
      %516 = stablehlo.select %455, %515, %514 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %517 = stablehlo.select %431, %453, %516 : tensor<1x197x3072xi1>, tensor<1x197x3072xf32>
      %518 = stablehlo.multiply %424, %517 : tensor<1x197x3072xf32>
      %519 = mhlo.bitcast %518 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x197x3072xf32>) -> tensor<197x3072xf32>
      mhlo.return %519 : tensor<197x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<197x3072xf32>, tensor<3072xf32>) -> tensor<197x3072xf32>
    %410 = stablehlo.dot_general %409, %arg190, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<197x3072xf32>, tensor<768x3072xf32>) -> tensor<197x768xf32>
    %411 = "mhlo.fusion"(%402, %410, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<197x768xf32>, %arg203: tensor<768xf32>):
      %420 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<197x768xf32>
      %421 = stablehlo.add %420, %arg202 : tensor<197x768xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<197x768xf32>) -> tensor<1x197x768xf32>
      %423 = stablehlo.add %422, %arg201 : tensor<1x197x768xf32>
      mhlo.return %423 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<197x768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %412 = "mhlo.fusion"(%411, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
      mhlo.return %420 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197xf32>
    %413 = "mhlo.fusion"(%411, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg202, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg201, %423 : tensor<1x197x768xf32>
      %425 = stablehlo.multiply %424, %424 : tensor<1x197x768xf32>
      mhlo.return %425 : tensor<1x197x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<1x197x768xf32>
    %414 = "mhlo.fusion"(%413, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x768xf32>, %arg202: tensor<f32>):
      %420 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %421 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %421 : tensor<f32>
      }) : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
      mhlo.return %420 : tensor<1x197x24xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x768xf32>, tensor<f32>) -> tensor<1x197x24xf32>
    %415 = "mhlo.fusion"(%414) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1x197x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %420 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x197x24xf32>, tensor<f32>) -> tensor<1x197xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<1x197xf32>
      %cst_3 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %424 = stablehlo.add %422, %423 : tensor<1x197xf32>
      %425 = stablehlo.rsqrt %424 : tensor<1x197xf32>
      mhlo.return %425 : tensor<1x197xf32>
    }) {output_operand_aliasing = []} : (tensor<1x197x24xf32>) -> tensor<1x197xf32>
    %416 = "mhlo.fusion"(%arg197, %arg196, %415, %411, %412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<1x197xf32>, %arg204: tensor<1x197x768xf32>, %arg205: tensor<1x197xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x197xf32>
      %421 = stablehlo.multiply %arg205, %420 : tensor<1x197xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %423 = stablehlo.broadcast_in_dim %422, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %424 = stablehlo.subtract %arg204, %423 : tensor<1x197x768xf32>
      %425 = mhlo.bitcast %arg203 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x197xf32>) -> tensor<197xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [1] : (tensor<197xf32>) -> tensor<1x197x768xf32>
      %427 = stablehlo.multiply %424, %426 : tensor<1x197x768xf32>
      %428 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<1x197x768xf32>
      %430 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<1x197x768xf32>
      %431 = stablehlo.add %429, %430 : tensor<1x197x768xf32>
      %432 = stablehlo.slice %431 [0:1, 0:1, 0:768] : (tensor<1x197x768xf32>) -> tensor<1x1x768xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1x768xf32>) -> tensor<768xf32>
      mhlo.return %433 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<1x197xf32>, tensor<1x197x768xf32>, tensor<1x197xf32>) -> tensor<768xf32>
    %417 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1000x768xf32>):
      %420 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,1000]{0,1}"} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
      %421 = mhlo.copy %420 : tensor<768x1000xf32>
      mhlo.return %421 : tensor<768x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000x768xf32>) -> tensor<768x1000xf32>
    %418 = "mhlo.fusion"(%arg199, %416, %417) <{fusion_kind = #mhlo<fusion_kind kOutput>}> ({
    ^bb0(%arg201: tensor<1000xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768x1000xf32>):
      %420 = stablehlo.dot %arg202, %arg203, precision = [DEFAULT, DEFAULT] : (tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
      %421 = stablehlo.add %arg201, %420 : tensor<1000xf32>
      mhlo.return %421 : tensor<1000xf32>
    }) {output_operand_aliasing = []} : (tensor<1000xf32>, tensor<768xf32>, tensor<768x1000xf32>) -> tensor<1000xf32>
    %419 = mhlo.bitcast %418 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %419 : tensor<1x1000xf32>
  }
}
