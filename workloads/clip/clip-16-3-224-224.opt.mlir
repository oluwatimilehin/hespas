module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_63.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_61.67(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_59.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_58.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_56.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_54.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_53.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_51.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_49.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_48.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_46.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_44.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_43.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_41.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_39.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_38.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_36.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_34.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_33.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_31.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_29.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_28.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_26.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_24.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.23(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_23.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.24(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_21.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.25(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_19.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_18.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_16.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.28(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_14.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.29(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_13.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_11.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.31(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_9.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_8.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_6.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.34(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_4.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @region_1.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.36(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    return %0 : tensor<16x50xf32>
  }
  func.func private @fused_computation.37(%arg0: tensor<16x2xf32>, %arg1: tensor<2xf32>) -> tensor<16x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<16x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16x2xf32>
    return %1 : tensor<16x2xf32>
  }
  func.func private @region_64.70.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.38(%arg0: tensor<16x2x768xf32>) -> tensor<16x768xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2x768xf32>, tensor<f32>) -> tensor<16x768xf32>
    %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x768xf32>
    return %2 : tensor<16x768xf32>
  }
  func.func private @fused_computation.39(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x49x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.slice %3 [0:16, 1:50, 0:768] : (tensor<16x50x768xf32>) -> tensor<16x49x768xf32>
    return %4 : tensor<16x49x768xf32>
  }
  func.func private @fused_computation.40(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.41(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.42(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.45(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.48(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.51(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.52(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.53(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.54(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.55(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.56(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.58(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.59(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.61(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.62(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.68(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.20.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.71(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.19.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.76(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.85(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.18.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.88(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.93(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.102(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.16.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.105(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.15.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.110(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.119(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.14.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.122(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.127(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.136(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.139(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.144(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.153(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.156(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.161(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.170(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.173(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.178(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.184(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.187(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.190(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.192(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.195(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.197(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.200(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.204(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.207(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.210(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.212(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<16x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<16x50x768xf32>
    return %2 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.219(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.220(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.221(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.224(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.225(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<16x50x768xf32>, %arg1: tensor<800x768xf32>, %arg2: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<800x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<800x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<16x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<16x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<16x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<16x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<16x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<16x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
    return %11 : tensor<800x3072xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50x768xf32>, %arg4: tensor<16x50xf32>) -> tensor<800x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<16x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<16x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %10 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.229(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.230(%arg0: tensor<16x50x768xf32>, %arg1: tensor<16x50xf32>) -> tensor<16x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x50x768xf32>
    return %4 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<16x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<16x50xf32>, %arg5: tensor<16x50xf32>, %arg6: tensor<16x50x768xf32>, %arg7: tensor<50x1x768xf32>, %arg8: tensor<1x50xi32>) -> tensor<16x50x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg8, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg8, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg8 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %arg7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %13 = stablehlo.select %10, %11, %12 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
    %16 = stablehlo.add %arg6, %15 : tensor<16x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %18 = stablehlo.multiply %arg5, %17 : tensor<16x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<16x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg4, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<16x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<16x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<16x50x768xf32>
    %27 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %28 = stablehlo.add %arg0, %27 : tensor<16x50x768xf32>
    %29 = stablehlo.add %26, %28 : tensor<16x50x768xf32>
    return %29 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.232(%arg0: tensor<768x768xf32>) -> tensor<16x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    return %2 : tensor<16x768x768xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<192x50x64xf32>) -> tensor<16x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<16x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
    return %3 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<16x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %5 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<16x12x50xf32>, %arg2: tensor<16x12x50xi1>) -> tensor<192x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<16x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
    return %6 : tensor<192x50x50xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<16x12x50xf32>, %arg1: tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<16x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<16x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<16x12x50x50xf32>
    return %5 : tensor<16x12x50x50xf32>
  }
  func.func private @fused_computation.237(%arg0: tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<16x12x50x50xi1>
    return %3 : tensor<16x12x50x50xi1>
  }
  func.func private @fused_computation.238(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<16x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
    return %7 : tensor<192x64x50xf32>
  }
  func.func private @fused_computation.239(%arg0: tensor<800x768xf32>, %arg1: tensor<768xf32>) -> tensor<192x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<800x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<800x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<16x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
    return %7 : tensor<192x50x64xf32>
  }
  func.func private @fused_computation.240(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<16x50xf32>, %arg7: tensor<16x50xf32>, %arg8: tensor<16x50x768xf32>, %arg9: tensor<50x1x768xf32>, %arg10: tensor<1x50xi32>) -> tensor<800x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg10, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg10, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg10 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %arg9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %13 = stablehlo.select %10, %11, %12 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
    %16 = stablehlo.add %arg8, %15 : tensor<16x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %18 = stablehlo.multiply %arg7, %17 : tensor<16x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<16x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<16x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<16x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<16x50x768xf32>
    %27 = stablehlo.multiply %arg3, %17 : tensor<16x50xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %29 = stablehlo.subtract %26, %28 : tensor<16x50x768xf32>
    %30 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<16x50x768xf32>
    %32 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<16x50x768xf32>
    %34 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %35 = stablehlo.add %33, %34 : tensor<16x50x768xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
    return %36 : tensor<800x768xf32>
  }
  func.func private @region_3.6.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.241(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.242(%arg0: tensor<16x50xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<16x50xf32>, %arg4: tensor<16x50xf32>, %arg5: tensor<16x50x768xf32>, %arg6: tensor<50x1x768xf32>, %arg7: tensor<1x50xi32>) -> tensor<16x50x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg7, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg7, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg7 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %arg6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %13 = stablehlo.select %10, %11, %12 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
    %16 = stablehlo.add %arg5, %15 : tensor<16x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %18 = stablehlo.multiply %arg4, %17 : tensor<16x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<16x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<16x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<16x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<16x50x768xf32>
    %27 = stablehlo.multiply %arg0, %17 : tensor<16x50xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %29 = stablehlo.subtract %26, %28 : tensor<16x50x768xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<16x50x768xf32>
    return %30 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.243(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x50xf32>, %arg3: tensor<16x50xf32>, %arg4: tensor<16x50x768xf32>, %arg5: tensor<50x1x768xf32>, %arg6: tensor<1x50xi32>) -> tensor<16x50x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg6, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg6, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg6 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %arg5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %13 = stablehlo.select %10, %11, %12 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
    %16 = stablehlo.add %arg4, %15 : tensor<16x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %18 = stablehlo.multiply %arg3, %17 : tensor<16x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<16x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<16x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<16x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<16x50x768xf32>
    return %26 : tensor<16x50x768xf32>
  }
  func.func private @region_3.6.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.244(%arg0: tensor<16x50x24xf32>) -> tensor<16x50xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<16x50xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %4 = stablehlo.add %2, %3 : tensor<16x50xf32>
    %5 = stablehlo.rsqrt %4 : tensor<16x50xf32>
    return %5 : tensor<16x50xf32>
  }
  func.func private @fused_computation.245(%arg0: tensor<16x50xf32>, %arg1: tensor<16x50x768xf32>, %arg2: tensor<50x1x768xf32>, %arg3: tensor<1x50xi32>) -> tensor<16x50x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg3, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg3, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg3 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %arg2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %13 = stablehlo.select %10, %11, %12 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
    %16 = stablehlo.add %arg1, %15 : tensor<16x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
    %18 = stablehlo.multiply %arg0, %17 : tensor<16x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<16x50x768xf32>
    %21 = stablehlo.multiply %20, %20 : tensor<16x50x768xf32>
    return %21 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.246(%arg0: tensor<16x50x768xf32>, %arg1: tensor<50x1x768xf32>, %arg2: tensor<1x50xi32>) -> tensor<16x50x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg2, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg2, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg2 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = stablehlo.compare GE, %4, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_1 = stablehlo.constant dense<49> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %7 = stablehlo.compare LE, %4, %6 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %8 = stablehlo.and %5, %7 : tensor<1x50xi1>
    %9 = mhlo.bitcast %8 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
    %11 = mhlo.bitcast %arg1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
    %13 = stablehlo.select %10, %11, %12 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
    %16 = stablehlo.add %arg0, %15 : tensor<16x50x768xf32>
    return %16 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.247(%arg0: tensor<50x768xf32>, %arg1: tensor<1x50xi32>) -> tensor<50x1x768xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %1 = stablehlo.compare LT, %arg1, %0 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x50xi32>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<1x50xi1>, tensor<1x50xi32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
    return %6 : tensor<50x1x768xf32>
  }
  func.func private @fused_computation.248(%arg0: tensor<16x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<16x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<16x1x768xf32>
    %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,768,7,7]{1,3,2,0}"} : (tensor<16x7x7x768xf32>) -> tensor<16x768x7x7xf32>
    %2 = mhlo.copy %1 : tensor<16x768x7x7xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x768x7x7xf32>) -> tensor<16x768x49xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,49,768]{1,2,0}"} : (tensor<16x768x49xf32>) -> tensor<16x49x768xf32>
    %5 = mhlo.copy %4 : tensor<16x49x768xf32>
    %6 = stablehlo.concatenate %0, %5, dim = 1 : (tensor<16x1x768xf32>, tensor<16x49x768xf32>) -> tensor<16x50x768xf32>
    return %6 : tensor<16x50x768xf32>
  }
  func.func private @fused_computation.249(%arg0: tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[32,32,3,768]{1,0,2,3}"} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
    %1 = mhlo.copy %0 : tensor<32x32x3x768xf32>
    return %1 : tensor<32x32x3x768xf32>
  }
  func.func private @fused_computation.250(%arg0: tensor<16x3x224x224xf32>) -> tensor<16x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,224,224,3]{2,1,3,0}"} : (tensor<16x3x224x224xf32>) -> tensor<16x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<16x224x224x3xf32>
    return %1 : tensor<16x224x224x3xf32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_5.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_5.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_7.12(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_7.12.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_10.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_10.16.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_12.18(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_12.18.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_15.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_15.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_17.23(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_17.23.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_20.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_20.26.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_22.28(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_22.28.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_25.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_25.31.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_27.33(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_27.33.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_30.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_30.36.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_32.38(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_32.38.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_35.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_35.41.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_37.43(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_37.43.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_40.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_40.46.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_42.48(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_42.48.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.15(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.33(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_45.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.34(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_45.51.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.16(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_47.53(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.35(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_47.53.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.17(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.36(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.37(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_50.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.38(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_50.56.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.18(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_52.58(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.39(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_52.58.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.19(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.40(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.41(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_55.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.42(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_55.61.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.20(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_57.63(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.43(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_57.63.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.21(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.44(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_3.6.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.45(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_60.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.46(%arg0: tensor<16x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<16x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    return %0 : tensor<16x12x50x2xf32>
  }
  func.func private @region_60.66.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.22(%arg0: tensor<16x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<16x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    return %0 : tensor<16x12x50xf32>
  }
  func.func private @region_62.68(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.47(%arg0: tensor<16x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<16x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    return %0 : tensor<16x12x50x2xi1>
  }
  func.func private @region_62.68.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.23(%arg0: tensor<16x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<16x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    return %0 : tensor<16x12x50xi1>
  }
  func.func private @region_3.6.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.48(%arg0: tensor<16x50x768xf32>, %arg1: tensor<f32>) -> tensor<16x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    return %0 : tensor<16x50x24xf32>
  }
  func.func private @region_64.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.49(%arg0: tensor<16x49x768xf32>, %arg1: tensor<f32>) -> tensor<16x2x768xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [7, 8], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 32, 1>, window_strides = array<i64: 1, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<16x49x768xf32>, tensor<f32>) -> tensor<16x2x768xf32>
    return %0 : tensor<16x2x768xf32>
  }
  func.func @main(%arg0: tensor<768xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<3072xf32>, %arg33: tensor<768x3072xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<3072x768xf32>, %arg48: tensor<3072xf32>, %arg49: tensor<768x3072xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<3072x768xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<768x3072xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<3072x768xf32>, %arg80: tensor<3072xf32>, %arg81: tensor<768x3072xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<3072x768xf32>, %arg96: tensor<3072xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<3072xf32>, %arg113: tensor<768x3072xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<3072x768xf32>, %arg128: tensor<3072xf32>, %arg129: tensor<768x3072xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768x768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<3072x768xf32>, %arg144: tensor<3072xf32>, %arg145: tensor<768x3072xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768x768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<3072x768xf32>, %arg160: tensor<3072xf32>, %arg161: tensor<768x3072xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<3072x768xf32>, %arg176: tensor<3072xf32>, %arg177: tensor<768x3072xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<768x3072xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<2x768xf32>, %arg198: tensor<2xf32>, %arg199: tensor<1x50xi32>, %arg200: tensor<16x3x224x224xf32>) -> tensor<16x2xf32> {
    %0 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x3x224x224xf32>):
      %435 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,224,224,3]{2,1,3,0}"} : (tensor<16x3x224x224xf32>) -> tensor<16x224x224x3xf32>
      %436 = mhlo.copy %435 : tensor<16x224x224x3xf32>
      mhlo.return %436 : tensor<16x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<16x3x224x224xf32>) -> tensor<16x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x32x32xf32>):
      %435 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[32,32,3,768]{1,0,2,3}"} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
      %436 = mhlo.copy %435 : tensor<32x32x3x768xf32>
      mhlo.return %436 : tensor<32x32x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [32, 32], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<16x224x224x3xf32>, tensor<32x32x3x768xf32>) -> tensor<16x7x7x768xf32>
    %3 = "mhlo.fusion"(%2, %arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x7x7x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x1x768xf32>
      %436 = stablehlo.transpose %arg201, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,768,7,7]{1,3,2,0}"} : (tensor<16x7x7x768xf32>) -> tensor<16x768x7x7xf32>
      %437 = mhlo.copy %436 : tensor<16x768x7x7xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x768x7x7xf32>) -> tensor<16x768x49xf32>
      %439 = stablehlo.transpose %438, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[16,49,768]{1,2,0}"} : (tensor<16x768x49xf32>) -> tensor<16x49x768xf32>
      %440 = mhlo.copy %439 : tensor<16x49x768xf32>
      %441 = stablehlo.concatenate %435, %440, dim = 1 : (tensor<16x1x768xf32>, tensor<16x49x768xf32>) -> tensor<16x50x768xf32>
      mhlo.return %441 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x7x7x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %4 = "mhlo.fusion"(%arg2, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg202, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg202, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg202 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %441 = "stablehlo.gather"(%arg201, %440) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      mhlo.return %441 : tensor<50x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<1x50xi32>) -> tensor<50x1x768xf32>
    %5 = "mhlo.fusion"(%3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<50x1x768xf32>, %arg203: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg203, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg203, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg203 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = stablehlo.compare GE, %439, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LE, %439, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %443 = stablehlo.and %440, %442 : tensor<1x50xi1>
      %444 = mhlo.bitcast %443 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %446 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %448 = stablehlo.select %445, %446, %447 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %449 = mhlo.bitcast %448 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
      %451 = stablehlo.add %arg201, %450 : tensor<16x50x768xf32>
      mhlo.return %451 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<16x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %7 = "mhlo.fusion"(%6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<50x1x768xf32>, %arg204: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg204, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg204, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg204 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = stablehlo.compare GE, %439, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LE, %439, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %443 = stablehlo.and %440, %442 : tensor<1x50xi1>
      %444 = mhlo.bitcast %443 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %446 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %448 = stablehlo.select %445, %446, %447 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %449 = mhlo.bitcast %448 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
      %451 = stablehlo.add %arg202, %450 : tensor<16x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %453 = stablehlo.multiply %arg201, %452 : tensor<16x50xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %455 = stablehlo.subtract %451, %454 : tensor<16x50x768xf32>
      %456 = stablehlo.multiply %455, %455 : tensor<16x50x768xf32>
      mhlo.return %456 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<16x50x768xf32>
    %8 = "mhlo.fusion"(%7, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %9 = "mhlo.fusion"(%8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %10 = "mhlo.fusion"(%arg4, %arg3, %9, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50xf32>, %arg205: tensor<16x50x768xf32>, %arg206: tensor<50x1x768xf32>, %arg207: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg207, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg207, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg207 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = stablehlo.compare GE, %439, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LE, %439, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %443 = stablehlo.and %440, %442 : tensor<1x50xi1>
      %444 = mhlo.bitcast %443 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %446 = mhlo.bitcast %arg206 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %448 = stablehlo.select %445, %446, %447 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %449 = mhlo.bitcast %448 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
      %451 = stablehlo.add %arg205, %450 : tensor<16x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %453 = stablehlo.multiply %arg204, %452 : tensor<16x50xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %455 = stablehlo.subtract %451, %454 : tensor<16x50x768xf32>
      %456 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<16x50x768xf32>
      %458 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %459 = stablehlo.multiply %457, %458 : tensor<16x50x768xf32>
      %460 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<16x50x768xf32>
      mhlo.return %461 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<16x50x768xf32>
    %11 = "mhlo.fusion"(%10, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %12 = "mhlo.fusion"(%11, %arg4, %arg3, %9, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<16x50xf32>, %arg205: tensor<16x50xf32>, %arg206: tensor<16x50x768xf32>, %arg207: tensor<50x1x768xf32>, %arg208: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg208, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg208, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg208 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = stablehlo.compare GE, %439, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LE, %439, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %443 = stablehlo.and %440, %442 : tensor<1x50xi1>
      %444 = mhlo.bitcast %443 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %446 = mhlo.bitcast %arg207 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %448 = stablehlo.select %445, %446, %447 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %449 = mhlo.bitcast %448 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
      %451 = stablehlo.add %arg206, %450 : tensor<16x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %453 = stablehlo.multiply %arg205, %452 : tensor<16x50xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %455 = stablehlo.subtract %451, %454 : tensor<16x50x768xf32>
      %456 = stablehlo.broadcast_in_dim %arg204, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<16x50x768xf32>
      %458 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %459 = stablehlo.multiply %457, %458 : tensor<16x50x768xf32>
      %460 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<16x50x768xf32>
      %462 = stablehlo.multiply %arg201, %452 : tensor<16x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %464 = stablehlo.subtract %461, %463 : tensor<16x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<16x50x768xf32>
      mhlo.return %465 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<16x50x768xf32>
    %13 = "mhlo.fusion"(%12, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %14 = "mhlo.fusion"(%13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %15 = "mhlo.fusion"(%arg14, %arg13, %14, %11, %arg4, %arg3, %9, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<16x50xf32>, %arg208: tensor<16x50xf32>, %arg209: tensor<16x50x768xf32>, %arg210: tensor<50x1x768xf32>, %arg211: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg211, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg211, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg211 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = stablehlo.compare GE, %439, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LE, %439, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %443 = stablehlo.and %440, %442 : tensor<1x50xi1>
      %444 = mhlo.bitcast %443 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %446 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %448 = stablehlo.select %445, %446, %447 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %449 = mhlo.bitcast %448 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
      %451 = stablehlo.add %arg209, %450 : tensor<16x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %453 = stablehlo.multiply %arg208, %452 : tensor<16x50xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %455 = stablehlo.subtract %451, %454 : tensor<16x50x768xf32>
      %456 = stablehlo.broadcast_in_dim %arg207, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<16x50x768xf32>
      %458 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %459 = stablehlo.multiply %457, %458 : tensor<16x50x768xf32>
      %460 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<16x50x768xf32>
      %462 = stablehlo.multiply %arg204, %452 : tensor<16x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %464 = stablehlo.subtract %461, %463 : tensor<16x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<16x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<16x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<16x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %471 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<800x768xf32>
    %16 = stablehlo.dot_general %15, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %17 = "mhlo.fusion"(%16, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %18 = stablehlo.dot_general %15, %arg5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %19 = "mhlo.fusion"(%18, %arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %20 = stablehlo.dot_general %17, %19, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %22 = "mhlo.fusion"(%21, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %23 = "mhlo.fusion"(%22, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %24 = "mhlo.fusion"(%23, %20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %25 = "mhlo.fusion"(%24, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %26 = "mhlo.fusion"(%20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %27 = "mhlo.fusion"(%26, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %28 = "mhlo.fusion"(%27, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %29 = "mhlo.fusion"(%24, %25, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %30 = stablehlo.dot_general %15, %arg7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %31 = "mhlo.fusion"(%30, %arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %32 = stablehlo.dot_general %29, %31, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %33 = "mhlo.fusion"(%32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %34 = "mhlo.fusion"(%arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %35 = stablehlo.dot_general %33, %34, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %36 = "mhlo.fusion"(%35, %arg12, %arg4, %arg3, %9, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<16x50xf32>, %arg206: tensor<16x50xf32>, %arg207: tensor<16x50x768xf32>, %arg208: tensor<50x1x768xf32>, %arg209: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %436 = stablehlo.compare LT, %arg209, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %438 = stablehlo.add %arg209, %437 : tensor<1x50xi32>
      %439 = stablehlo.select %436, %438, %arg209 : tensor<1x50xi1>, tensor<1x50xi32>
      %440 = stablehlo.compare GE, %439, %435 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %441 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %442 = stablehlo.compare LE, %439, %441 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %443 = stablehlo.and %440, %442 : tensor<1x50xi1>
      %444 = mhlo.bitcast %443 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %446 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %447 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %448 = stablehlo.select %445, %446, %447 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %449 = mhlo.bitcast %448 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %450 = stablehlo.broadcast_in_dim %449, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<16x50x768xf32>
      %451 = stablehlo.add %arg207, %450 : tensor<16x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %453 = stablehlo.multiply %arg206, %452 : tensor<16x50xf32>
      %454 = stablehlo.broadcast_in_dim %453, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %455 = stablehlo.subtract %451, %454 : tensor<16x50x768xf32>
      %456 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %457 = stablehlo.multiply %455, %456 : tensor<16x50x768xf32>
      %458 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %459 = stablehlo.multiply %457, %458 : tensor<16x50x768xf32>
      %460 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<16x50x768xf32>
      %462 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<16x50x768xf32>
      %464 = stablehlo.add %461, %463 : tensor<16x50x768xf32>
      mhlo.return %464 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<16x50x768xf32>
    %37 = "mhlo.fusion"(%36, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %38 = "mhlo.fusion"(%36, %37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %39 = "mhlo.fusion"(%38, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %40 = "mhlo.fusion"(%39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %41 = "mhlo.fusion"(%arg20, %arg19, %40, %36, %37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %42 = stablehlo.dot_general %41, %arg15, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %43 = "mhlo.fusion"(%42, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %44 = stablehlo.dot_general %43, %arg17, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %45 = "mhlo.fusion"(%36, %44, %arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %46 = "mhlo.fusion"(%45, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %47 = "mhlo.fusion"(%45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %48 = "mhlo.fusion"(%47, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %49 = "mhlo.fusion"(%48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %50 = "mhlo.fusion"(%arg30, %arg29, %49, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %51 = stablehlo.dot_general %50, %arg25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %52 = "mhlo.fusion"(%51, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %53 = stablehlo.dot_general %50, %arg21, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %54 = "mhlo.fusion"(%53, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %55 = stablehlo.dot_general %52, %54, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %56 = mhlo.bitcast %55 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %57 = "mhlo.fusion"(%56, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %58 = "mhlo.fusion"(%57, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %59 = "mhlo.fusion"(%58, %55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %60 = "mhlo.fusion"(%59, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %61 = "mhlo.fusion"(%55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %62 = "mhlo.fusion"(%61, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %63 = "mhlo.fusion"(%62, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %64 = "mhlo.fusion"(%59, %60, %63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %65 = stablehlo.dot_general %50, %arg23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %66 = "mhlo.fusion"(%65, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %67 = stablehlo.dot_general %64, %66, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %69 = "mhlo.fusion"(%arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %70 = stablehlo.dot_general %68, %69, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %71 = "mhlo.fusion"(%45, %70, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %72 = "mhlo.fusion"(%71, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %74 = "mhlo.fusion"(%73, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %75 = "mhlo.fusion"(%74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %76 = "mhlo.fusion"(%arg36, %arg35, %75, %71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %77 = stablehlo.dot_general %76, %arg31, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %78 = "mhlo.fusion"(%77, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %79 = stablehlo.dot_general %78, %arg33, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %80 = "mhlo.fusion"(%71, %79, %arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %81 = "mhlo.fusion"(%80, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %82 = "mhlo.fusion"(%80, %81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %83 = "mhlo.fusion"(%82, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %84 = "mhlo.fusion"(%83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %85 = "mhlo.fusion"(%arg46, %arg45, %84, %80, %81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %86 = stablehlo.dot_general %85, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %87 = "mhlo.fusion"(%86, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %88 = stablehlo.dot_general %85, %arg37, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %89 = "mhlo.fusion"(%88, %arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %90 = stablehlo.dot_general %87, %89, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %92 = "mhlo.fusion"(%91, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %93 = "mhlo.fusion"(%92, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %94 = "mhlo.fusion"(%93, %90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %95 = "mhlo.fusion"(%94, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %96 = "mhlo.fusion"(%90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %97 = "mhlo.fusion"(%96, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %98 = "mhlo.fusion"(%97, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %99 = "mhlo.fusion"(%94, %95, %98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %100 = stablehlo.dot_general %85, %arg39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %101 = "mhlo.fusion"(%100, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %102 = stablehlo.dot_general %99, %101, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %103 = "mhlo.fusion"(%102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %104 = "mhlo.fusion"(%arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %105 = stablehlo.dot_general %103, %104, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %106 = "mhlo.fusion"(%80, %105, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %108 = "mhlo.fusion"(%106, %107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %109 = "mhlo.fusion"(%108, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %110 = "mhlo.fusion"(%109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %111 = "mhlo.fusion"(%arg52, %arg51, %110, %106, %107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %112 = stablehlo.dot_general %111, %arg47, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %113 = "mhlo.fusion"(%112, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %114 = stablehlo.dot_general %113, %arg49, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %115 = "mhlo.fusion"(%106, %114, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %116 = "mhlo.fusion"(%115, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %117 = "mhlo.fusion"(%115, %116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %118 = "mhlo.fusion"(%117, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %120 = "mhlo.fusion"(%arg62, %arg61, %119, %115, %116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %121 = stablehlo.dot_general %120, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %122 = "mhlo.fusion"(%121, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %123 = stablehlo.dot_general %120, %arg53, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %124 = "mhlo.fusion"(%123, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %125 = stablehlo.dot_general %122, %124, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %126 = mhlo.bitcast %125 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %127 = "mhlo.fusion"(%126, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %128 = "mhlo.fusion"(%127, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %129 = "mhlo.fusion"(%128, %125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %130 = "mhlo.fusion"(%129, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %131 = "mhlo.fusion"(%125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %132 = "mhlo.fusion"(%131, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %133 = "mhlo.fusion"(%132, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %134 = "mhlo.fusion"(%129, %130, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %135 = stablehlo.dot_general %120, %arg55, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %136 = "mhlo.fusion"(%135, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %137 = stablehlo.dot_general %134, %136, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %138 = "mhlo.fusion"(%137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %139 = "mhlo.fusion"(%arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %140 = stablehlo.dot_general %138, %139, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %141 = "mhlo.fusion"(%115, %140, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %143 = "mhlo.fusion"(%141, %142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %144 = "mhlo.fusion"(%143, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %146 = "mhlo.fusion"(%arg68, %arg67, %145, %141, %142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %147 = stablehlo.dot_general %146, %arg63, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %148 = "mhlo.fusion"(%147, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %149 = stablehlo.dot_general %148, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %150 = "mhlo.fusion"(%141, %149, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %151 = "mhlo.fusion"(%150, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %152 = "mhlo.fusion"(%150, %151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %153 = "mhlo.fusion"(%152, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %155 = "mhlo.fusion"(%arg78, %arg77, %154, %150, %151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %156 = stablehlo.dot_general %155, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %157 = "mhlo.fusion"(%156, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %158 = stablehlo.dot_general %155, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %159 = "mhlo.fusion"(%158, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %160 = stablehlo.dot_general %157, %159, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %161 = mhlo.bitcast %160 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %162 = "mhlo.fusion"(%161, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %163 = "mhlo.fusion"(%162, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %164 = "mhlo.fusion"(%163, %160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %165 = "mhlo.fusion"(%164, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %166 = "mhlo.fusion"(%160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %167 = "mhlo.fusion"(%166, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %168 = "mhlo.fusion"(%167, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %169 = "mhlo.fusion"(%164, %165, %168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %170 = stablehlo.dot_general %155, %arg71, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %171 = "mhlo.fusion"(%170, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %172 = stablehlo.dot_general %169, %171, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %173 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %174 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %175 = stablehlo.dot_general %173, %174, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %176 = "mhlo.fusion"(%150, %175, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %177 = "mhlo.fusion"(%176, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %178 = "mhlo.fusion"(%176, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %179 = "mhlo.fusion"(%178, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %180 = "mhlo.fusion"(%179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %181 = "mhlo.fusion"(%arg84, %arg83, %180, %176, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %182 = stablehlo.dot_general %181, %arg79, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %183 = "mhlo.fusion"(%182, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %184 = stablehlo.dot_general %183, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %185 = "mhlo.fusion"(%176, %184, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %186 = "mhlo.fusion"(%185, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %187 = "mhlo.fusion"(%185, %186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %188 = "mhlo.fusion"(%187, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %190 = "mhlo.fusion"(%arg94, %arg93, %189, %185, %186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %191 = stablehlo.dot_general %190, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %192 = "mhlo.fusion"(%191, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %193 = stablehlo.dot_general %190, %arg85, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %194 = "mhlo.fusion"(%193, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %195 = stablehlo.dot_general %192, %194, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %196 = mhlo.bitcast %195 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %197 = "mhlo.fusion"(%196, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %198 = "mhlo.fusion"(%197, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %199 = "mhlo.fusion"(%198, %195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %200 = "mhlo.fusion"(%199, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %201 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %202 = "mhlo.fusion"(%201, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %203 = "mhlo.fusion"(%202, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %204 = "mhlo.fusion"(%199, %200, %203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %205 = stablehlo.dot_general %190, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %206 = "mhlo.fusion"(%205, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %207 = stablehlo.dot_general %204, %206, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %209 = "mhlo.fusion"(%arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %210 = stablehlo.dot_general %208, %209, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %211 = "mhlo.fusion"(%185, %210, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %212 = "mhlo.fusion"(%211, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %213 = "mhlo.fusion"(%211, %212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %214 = "mhlo.fusion"(%213, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %216 = "mhlo.fusion"(%arg100, %arg99, %215, %211, %212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %217 = stablehlo.dot_general %216, %arg95, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %218 = "mhlo.fusion"(%217, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %219 = stablehlo.dot_general %218, %arg97, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %220 = "mhlo.fusion"(%211, %219, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %221 = "mhlo.fusion"(%220, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %222 = "mhlo.fusion"(%220, %221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %223 = "mhlo.fusion"(%222, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %224 = "mhlo.fusion"(%223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %225 = "mhlo.fusion"(%arg110, %arg109, %224, %220, %221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %226 = stablehlo.dot_general %225, %arg105, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %227 = "mhlo.fusion"(%226, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %228 = stablehlo.dot_general %225, %arg101, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %229 = "mhlo.fusion"(%228, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %230 = stablehlo.dot_general %227, %229, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %231 = mhlo.bitcast %230 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %232 = "mhlo.fusion"(%231, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %233 = "mhlo.fusion"(%232, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %234 = "mhlo.fusion"(%233, %230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %235 = "mhlo.fusion"(%234, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %236 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %237 = "mhlo.fusion"(%236, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %238 = "mhlo.fusion"(%237, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %239 = "mhlo.fusion"(%234, %235, %238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %240 = stablehlo.dot_general %225, %arg103, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %241 = "mhlo.fusion"(%240, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %242 = stablehlo.dot_general %239, %241, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %243 = "mhlo.fusion"(%242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %244 = "mhlo.fusion"(%arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %245 = stablehlo.dot_general %243, %244, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %246 = "mhlo.fusion"(%220, %245, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %247 = "mhlo.fusion"(%246, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %248 = "mhlo.fusion"(%246, %247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %249 = "mhlo.fusion"(%248, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %251 = "mhlo.fusion"(%arg116, %arg115, %250, %246, %247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %252 = stablehlo.dot_general %251, %arg111, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %253 = "mhlo.fusion"(%252, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %254 = stablehlo.dot_general %253, %arg113, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %255 = "mhlo.fusion"(%246, %254, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %256 = "mhlo.fusion"(%255, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %257 = "mhlo.fusion"(%255, %256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %258 = "mhlo.fusion"(%257, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %260 = "mhlo.fusion"(%arg126, %arg125, %259, %255, %256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %261 = stablehlo.dot_general %260, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %262 = "mhlo.fusion"(%261, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %263 = stablehlo.dot_general %260, %arg117, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %264 = "mhlo.fusion"(%263, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %265 = stablehlo.dot_general %262, %264, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %266 = mhlo.bitcast %265 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %267 = "mhlo.fusion"(%266, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %268 = "mhlo.fusion"(%267, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %269 = "mhlo.fusion"(%268, %265) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %270 = "mhlo.fusion"(%269, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %271 = "mhlo.fusion"(%265) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %272 = "mhlo.fusion"(%271, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %273 = "mhlo.fusion"(%272, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %274 = "mhlo.fusion"(%269, %270, %273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %275 = stablehlo.dot_general %260, %arg119, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %276 = "mhlo.fusion"(%275, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %277 = stablehlo.dot_general %274, %276, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %278 = "mhlo.fusion"(%277) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %279 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %280 = stablehlo.dot_general %278, %279, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %281 = "mhlo.fusion"(%255, %280, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %282 = "mhlo.fusion"(%281, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %283 = "mhlo.fusion"(%281, %282) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %284 = "mhlo.fusion"(%283, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %285 = "mhlo.fusion"(%284) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %286 = "mhlo.fusion"(%arg132, %arg131, %285, %281, %282) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %287 = stablehlo.dot_general %286, %arg127, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %288 = "mhlo.fusion"(%287, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %289 = stablehlo.dot_general %288, %arg129, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %290 = "mhlo.fusion"(%281, %289, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %291 = "mhlo.fusion"(%290, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %292 = "mhlo.fusion"(%290, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %293 = "mhlo.fusion"(%292, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %294 = "mhlo.fusion"(%293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %295 = "mhlo.fusion"(%arg142, %arg141, %294, %290, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %296 = stablehlo.dot_general %295, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %297 = "mhlo.fusion"(%296, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %298 = stablehlo.dot_general %295, %arg133, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %299 = "mhlo.fusion"(%298, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %300 = stablehlo.dot_general %297, %299, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %301 = mhlo.bitcast %300 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %302 = "mhlo.fusion"(%301, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %303 = "mhlo.fusion"(%302, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %304 = "mhlo.fusion"(%303, %300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %305 = "mhlo.fusion"(%304, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %306 = "mhlo.fusion"(%300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %307 = "mhlo.fusion"(%306, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %308 = "mhlo.fusion"(%307, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %309 = "mhlo.fusion"(%304, %305, %308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %310 = stablehlo.dot_general %295, %arg135, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %311 = "mhlo.fusion"(%310, %arg136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %312 = stablehlo.dot_general %309, %311, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %313 = "mhlo.fusion"(%312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %314 = "mhlo.fusion"(%arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %315 = stablehlo.dot_general %313, %314, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %316 = "mhlo.fusion"(%290, %315, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %317 = "mhlo.fusion"(%316, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %318 = "mhlo.fusion"(%316, %317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %319 = "mhlo.fusion"(%318, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %320 = "mhlo.fusion"(%319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %321 = "mhlo.fusion"(%arg148, %arg147, %320, %316, %317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %322 = stablehlo.dot_general %321, %arg143, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %323 = "mhlo.fusion"(%322, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %324 = stablehlo.dot_general %323, %arg145, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %325 = "mhlo.fusion"(%316, %324, %arg146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %326 = "mhlo.fusion"(%325, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %327 = "mhlo.fusion"(%325, %326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %328 = "mhlo.fusion"(%327, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %329 = "mhlo.fusion"(%328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %330 = "mhlo.fusion"(%arg158, %arg157, %329, %325, %326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %331 = stablehlo.dot_general %330, %arg153, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %332 = "mhlo.fusion"(%331, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %333 = stablehlo.dot_general %330, %arg149, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %334 = "mhlo.fusion"(%333, %arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %335 = stablehlo.dot_general %332, %334, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %336 = mhlo.bitcast %335 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %337 = "mhlo.fusion"(%336, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %338 = "mhlo.fusion"(%337, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %339 = "mhlo.fusion"(%338, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %340 = "mhlo.fusion"(%339, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %341 = "mhlo.fusion"(%335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %342 = "mhlo.fusion"(%341, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %343 = "mhlo.fusion"(%342, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %344 = "mhlo.fusion"(%339, %340, %343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %345 = stablehlo.dot_general %330, %arg151, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %346 = "mhlo.fusion"(%345, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %347 = stablehlo.dot_general %344, %346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %348 = "mhlo.fusion"(%347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %349 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %350 = stablehlo.dot_general %348, %349, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %351 = "mhlo.fusion"(%325, %350, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %352 = "mhlo.fusion"(%351, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %353 = "mhlo.fusion"(%351, %352) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %354 = "mhlo.fusion"(%353, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %355 = "mhlo.fusion"(%354) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %356 = "mhlo.fusion"(%arg164, %arg163, %355, %351, %352) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %357 = stablehlo.dot_general %356, %arg159, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %358 = "mhlo.fusion"(%357, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %359 = stablehlo.dot_general %358, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %360 = "mhlo.fusion"(%351, %359, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %361 = "mhlo.fusion"(%360, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %362 = "mhlo.fusion"(%360, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %363 = "mhlo.fusion"(%362, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %364 = "mhlo.fusion"(%363) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %365 = "mhlo.fusion"(%arg174, %arg173, %364, %360, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %366 = stablehlo.dot_general %365, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %367 = "mhlo.fusion"(%366, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %368 = stablehlo.dot_general %365, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %369 = "mhlo.fusion"(%368, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %370 = stablehlo.dot_general %367, %369, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %371 = mhlo.bitcast %370 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %372 = "mhlo.fusion"(%371, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %373 = "mhlo.fusion"(%372, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %374 = "mhlo.fusion"(%373, %370) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %375 = "mhlo.fusion"(%374, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %376 = "mhlo.fusion"(%370) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %377 = "mhlo.fusion"(%376, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %378 = "mhlo.fusion"(%377, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %379 = "mhlo.fusion"(%374, %375, %378) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %380 = stablehlo.dot_general %365, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %381 = "mhlo.fusion"(%380, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %382 = stablehlo.dot_general %379, %381, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %383 = "mhlo.fusion"(%382) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %384 = "mhlo.fusion"(%arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %385 = stablehlo.dot_general %383, %384, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %386 = "mhlo.fusion"(%360, %385, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %387 = "mhlo.fusion"(%386, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %388 = "mhlo.fusion"(%386, %387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %389 = "mhlo.fusion"(%388, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %390 = "mhlo.fusion"(%389) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %391 = "mhlo.fusion"(%arg180, %arg179, %390, %386, %387) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %392 = stablehlo.dot_general %391, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %393 = "mhlo.fusion"(%392, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %394 = stablehlo.dot_general %393, %arg177, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %395 = "mhlo.fusion"(%386, %394, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %396 = "mhlo.fusion"(%395, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %397 = "mhlo.fusion"(%395, %396) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %398 = "mhlo.fusion"(%397, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %399 = "mhlo.fusion"(%398) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %400 = "mhlo.fusion"(%arg190, %arg189, %399, %395, %396) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %401 = stablehlo.dot_general %400, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %402 = "mhlo.fusion"(%401, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %441 = mhlo.copy %440 : tensor<16x12x50x64xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %442 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %403 = stablehlo.dot_general %400, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %404 = "mhlo.fusion"(%403, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<800x768xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<800x768xf32>
      %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %440 = stablehlo.transpose %439, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,64,50]{2,1,3,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x64x50xf32>
      %441 = mhlo.copy %440 : tensor<16x12x64x50xf32>
      %442 = mhlo.bitcast %441 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x64x50xf32>) -> tensor<192x64x50xf32>
      mhlo.return %442 : tensor<192x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x64x50xf32>
    %405 = stablehlo.dot_general %402, %404, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x64xf32>, tensor<192x64x50xf32>) -> tensor<192x50x50xf32>
    %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %407 = "mhlo.fusion"(%406, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
      mhlo.return %435 : tensor<16x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50x2xf32>
    %408 = "mhlo.fusion"(%407, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %409 = "mhlo.fusion"(%408, %405) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50xf32>, %arg202: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50xf32>
      %437 = stablehlo.maximum %436, %arg201 : tensor<16x12x50xf32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.subtract %435, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.exponential %439 : tensor<16x12x50x50xf32>
      mhlo.return %440 : tensor<16x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50xf32>, tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
    %410 = "mhlo.fusion"(%409, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
      mhlo.return %435 : tensor<16x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<f32>) -> tensor<16x12x50xf32>
    %411 = "mhlo.fusion"(%405) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x50xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %437 = stablehlo.compare EQ, %435, %436 : (tensor<16x12x50x50xf32>, tensor<16x12x50x50xf32>) -> tensor<16x12x50x50xi1>
      %438 = stablehlo.not %437 : tensor<16x12x50x50xi1>
      mhlo.return %438 : tensor<16x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<192x50x50xf32>) -> tensor<16x12x50x50xi1>
    %412 = "mhlo.fusion"(%411, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xi1>, %arg202: tensor<i1>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %436 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %436 : tensor<i1>
      }) : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
      mhlo.return %435 : tensor<16x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xi1>, tensor<i1>) -> tensor<16x12x50x2xi1>
    %413 = "mhlo.fusion"(%412, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x2xi1>, %arg202: tensor<i1>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
      mhlo.return %435 : tensor<16x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x2xi1>, tensor<i1>) -> tensor<16x12x50xi1>
    %414 = "mhlo.fusion"(%409, %410, %413) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x12x50x50xf32>, %arg202: tensor<16x12x50xf32>, %arg203: tensor<16x12x50xi1>):
      %435 = stablehlo.not %arg203 : tensor<16x12x50xi1>
      %436 = stablehlo.broadcast_in_dim %435, dims = [0, 1, 2] : (tensor<16x12x50xi1>) -> tensor<16x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x12x50x50xf32>
      %438 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<16x12x50xf32>) -> tensor<16x12x50x50xf32>
      %439 = stablehlo.divide %arg201, %438 : tensor<16x12x50x50xf32>
      %440 = stablehlo.select %436, %437, %439 : tensor<16x12x50x50xi1>, tensor<16x12x50x50xf32>
      %441 = mhlo.bitcast %440 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x50xf32>) -> tensor<192x50x50xf32>
      mhlo.return %441 : tensor<192x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x12x50x50xf32>, tensor<16x12x50xf32>, tensor<16x12x50xi1>) -> tensor<192x50x50xf32>
    %415 = stablehlo.dot_general %400, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<768x768xf32>) -> tensor<800x768xf32>
    %416 = "mhlo.fusion"(%415, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x768xf32>, %arg202: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x12x64xf32>
      %438 = stablehlo.transpose %437, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,12,50,64]{3,1,2,0}"} : (tensor<16x50x12x64xf32>) -> tensor<16x12x50x64xf32>
      %439 = mhlo.copy %438 : tensor<16x12x50x64xf32>
      %440 = mhlo.bitcast %439 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x12x50x64xf32>) -> tensor<192x50x64xf32>
      mhlo.return %440 : tensor<192x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<800x768xf32>, tensor<768xf32>) -> tensor<192x50x64xf32>
    %417 = stablehlo.dot_general %414, %416, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<192x50x50xf32>, tensor<192x50x64xf32>) -> tensor<192x50x64xf32>
    %418 = "mhlo.fusion"(%417) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x50x64xf32>):
      %435 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<192x50x64xf32>) -> tensor<16x12x50x64xf32>
      %436 = stablehlo.transpose %435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[16,50,12,64]{3,1,2,0}"} : (tensor<16x12x50x64xf32>) -> tensor<16x50x12x64xf32>
      %437 = mhlo.copy %436 : tensor<16x50x12x64xf32>
      %438 = mhlo.bitcast %437 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x50x12x64xf32>) -> tensor<16x50x768xf32>
      mhlo.return %438 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<192x50x64xf32>) -> tensor<16x50x768xf32>
    %419 = "mhlo.fusion"(%arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %435 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %436 = mhlo.copy %435 : tensor<768x768xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
      mhlo.return %437 : tensor<16x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<16x768x768xf32>
    %420 = stablehlo.dot_general %418, %419, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x50x768xf32>, tensor<16x768x768xf32>) -> tensor<16x50x768xf32>
    %421 = "mhlo.fusion"(%395, %420, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %436 = stablehlo.add %arg202, %435 : tensor<16x50x768xf32>
      %437 = stablehlo.add %arg201, %436 : tensor<16x50x768xf32>
      mhlo.return %437 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50x768xf32>, tensor<768xf32>) -> tensor<16x50x768xf32>
    %422 = "mhlo.fusion"(%421, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
      mhlo.return %435 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50xf32>
    %423 = "mhlo.fusion"(%421, %422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg202, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.multiply %438, %438 : tensor<16x50x768xf32>
      mhlo.return %439 : tensor<16x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<16x50x768xf32>
    %424 = "mhlo.fusion"(%423, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
      mhlo.return %435 : tensor<16x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<f32>) -> tensor<16x50x24xf32>
    %425 = "mhlo.fusion"(%424) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x24xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<16x50x24xf32>, tensor<f32>) -> tensor<16x50xf32>
      %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x50xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %439 = stablehlo.add %437, %438 : tensor<16x50xf32>
      %440 = stablehlo.rsqrt %439 : tensor<16x50xf32>
      mhlo.return %440 : tensor<16x50xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x24xf32>) -> tensor<16x50xf32>
    %426 = "mhlo.fusion"(%arg196, %arg195, %425, %421, %422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<16x50xf32>, %arg204: tensor<16x50x768xf32>, %arg205: tensor<16x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50xf32>
      %436 = stablehlo.multiply %arg205, %435 : tensor<16x50xf32>
      %437 = stablehlo.broadcast_in_dim %436, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.subtract %arg204, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<16x50xf32>) -> tensor<16x50x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<16x50x768xf32>
      %441 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %442 = stablehlo.multiply %440, %441 : tensor<16x50x768xf32>
      %443 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<16x50x768xf32>
      %444 = stablehlo.add %442, %443 : tensor<16x50x768xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x768xf32>) -> tensor<800x768xf32>
      mhlo.return %445 : tensor<800x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x50xf32>, tensor<16x50x768xf32>, tensor<16x50xf32>) -> tensor<800x768xf32>
    %427 = stablehlo.dot_general %426, %arg191, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x768xf32>, tensor<3072x768xf32>) -> tensor<800x3072xf32>
    %428 = "mhlo.fusion"(%427, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<800x3072xf32>, %arg202: tensor<3072xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<800x3072xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<800x3072xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x3072xf32>) -> tensor<16x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %439 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x50x3072xf32>
      %440 = stablehlo.multiply %437, %439 : tensor<16x50x3072xf32>
      %441 = stablehlo.negate %440 : tensor<16x50x3072xf32>
      %442 = stablehlo.exponential %441 : tensor<16x50x3072xf32>
      %443 = stablehlo.add %438, %442 : tensor<16x50x3072xf32>
      %444 = stablehlo.divide %438, %443 : tensor<16x50x3072xf32>
      %445 = stablehlo.multiply %437, %444 : tensor<16x50x3072xf32>
      %446 = mhlo.bitcast %445 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<16x50x3072xf32>) -> tensor<800x3072xf32>
      mhlo.return %446 : tensor<800x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<800x3072xf32>, tensor<3072xf32>) -> tensor<800x3072xf32>
    %429 = stablehlo.dot_general %428, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<800x3072xf32>, tensor<768x3072xf32>) -> tensor<800x768xf32>
    %430 = "mhlo.fusion"(%421, %429, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x50x768xf32>, %arg202: tensor<800x768xf32>, %arg203: tensor<768xf32>):
      %435 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<800x768xf32>
      %436 = stablehlo.add %435, %arg202 : tensor<800x768xf32>
      %437 = mhlo.bitcast %436 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<800x768xf32>) -> tensor<16x50x768xf32>
      %438 = stablehlo.add %arg201, %437 : tensor<16x50x768xf32>
      %439 = stablehlo.slice %438 [0:16, 1:50, 0:768] : (tensor<16x50x768xf32>) -> tensor<16x49x768xf32>
      mhlo.return %439 : tensor<16x49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x50x768xf32>, tensor<800x768xf32>, tensor<768xf32>) -> tensor<16x49x768xf32>
    %431 = "mhlo.fusion"(%430, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x49x768xf32>, %arg202: tensor<f32>):
      %435 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [7, 8], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 32, 1>, window_strides = array<i64: 1, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %436 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %436 : tensor<f32>
      }) : (tensor<16x49x768xf32>, tensor<f32>) -> tensor<16x2x768xf32>
      mhlo.return %435 : tensor<16x2x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x49x768xf32>, tensor<f32>) -> tensor<16x2x768xf32>
    %432 = "mhlo.fusion"(%431) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x2x768xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %435 = stablehlo.reduce(%arg201 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<16x2x768xf32>, tensor<f32>) -> tensor<16x768xf32>
      %cst_2 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x768xf32>
      %437 = stablehlo.multiply %435, %436 : tensor<16x768xf32>
      mhlo.return %437 : tensor<16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2x768xf32>) -> tensor<16x768xf32>
    %433 = stablehlo.dot_general %432, %arg197, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x768xf32>, tensor<2x768xf32>) -> tensor<16x2xf32>
    %434 = "mhlo.fusion"(%433, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<16x2xf32>, %arg202: tensor<2xf32>):
      %435 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<2xf32>) -> tensor<16x2xf32>
      %436 = stablehlo.add %435, %arg201 : tensor<16x2xf32>
      mhlo.return %436 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>, tensor<2xf32>) -> tensor<16x2xf32>
    return %434 : tensor<16x2xf32>
  }
}
