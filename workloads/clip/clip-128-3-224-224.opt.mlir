module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_64.70.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<128x2x768xf32>, %arg1: tensor<f32>) -> tensor<128x768xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<128x2x768xf32>, tensor<f32>) -> tensor<128x768xf32>
    return %0 : tensor<128x768xf32>
  }
  func.func private @region_3.6.clone.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_63.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_61.67(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_60.66.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_59.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_58.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_56.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_55.61.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.20.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_54.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.19.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_53.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_51.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_50.56.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.18.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_49.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_48.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_46.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_45.51.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.16.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.23(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_44.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.24(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.15.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.25(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_43.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_41.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_40.46.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.28(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.14.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.29(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_39.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.31(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_38.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_36.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_35.41.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.34(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_34.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.36(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.37(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_33.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.38(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_31.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.39(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_30.36.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.40(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.41(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_29.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.42(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.43(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_28.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.44(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_26.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_25.31.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.46(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.47(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_24.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.48(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.49(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_23.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.50(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_21.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.51(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_20.26.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.52(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.53(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_19.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.54(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.55(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_18.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.56(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_16.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.57(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_15.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.58(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.59(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_14.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.60(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.61(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_13.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.62(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_11.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.63(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_10.16.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.64(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.65(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_9.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.66(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.67(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_8.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.68(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_6.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.69(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_5.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.70(%arg0: tensor<128x12x50x2xf32>, %arg1: tensor<f32>) -> tensor<128x12x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    return %0 : tensor<128x12x50xf32>
  }
  func.func private @region_3.6.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.71(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_4.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.72(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_3.6.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.73(%arg0: tensor<128x50x24xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @region_1.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.74(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    return %0 : tensor<128x50xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<128x2xf32>, %arg1: tensor<2xf32>) -> tensor<128x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<128x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<128x2xf32>
    return %1 : tensor<128x2xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<128x768xf32>) -> tensor<128x768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x768xf32>
    return %1 : tensor<128x768xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x49x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.slice %3 [0:128, 1:50, 0:768] : (tensor<128x50x768xf32>) -> tensor<128x49x768xf32>
    return %4 : tensor<128x49x768xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.89(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.106(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.123(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.140(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.157(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.173(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.174(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.184(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.191(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.192(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.195(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.197(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.200(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.207(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.208(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.210(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.219(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.220(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.221(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.225(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.229(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.230(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.232(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.237(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.238(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.239(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.240(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.241(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.242(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.243(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.244(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.245(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.246(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.247(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.248(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.249(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.250(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.251(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.252(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<128x50x768xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x50x768xf32>
    return %2 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.253(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.254(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.255(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.256(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.257(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.258(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.259(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.260(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.261(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.262(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.263(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.264(%arg0: tensor<128x50x768xf32>, %arg1: tensor<6400x768xf32>, %arg2: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.265(%arg0: tensor<6400x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<6400x3072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x3072xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %cst_0 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
    %5 = stablehlo.multiply %2, %4 : tensor<128x50x3072xf32>
    %6 = stablehlo.negate %5 : tensor<128x50x3072xf32>
    %7 = stablehlo.exponential %6 : tensor<128x50x3072xf32>
    %8 = stablehlo.add %3, %7 : tensor<128x50x3072xf32>
    %9 = stablehlo.divide %3, %8 : tensor<128x50x3072xf32>
    %10 = stablehlo.multiply %2, %9 : tensor<128x50x3072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
    return %11 : tensor<6400x3072xf32>
  }
  func.func private @fused_computation.266(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50x768xf32>, %arg4: tensor<128x50xf32>) -> tensor<6400x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<128x50x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128x50x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %9 = stablehlo.add %7, %8 : tensor<128x50x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %10 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.267(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.268(%arg0: tensor<128x50x768xf32>, %arg1: tensor<128x50xf32>) -> tensor<128x50x768xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x50xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<128x50x768xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<128x50x768xf32>
    return %4 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.269(%arg0: tensor<128x50x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<128x50xf32>, %arg5: tensor<128x50xf32>, %arg6: tensor<128x50x768xf32>, %arg7: tensor<50x1x768xf32>, %arg8: tensor<1x50xi32>) -> tensor<128x50x768xf32> {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
    %16 = stablehlo.add %arg6, %15 : tensor<128x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %18 = stablehlo.multiply %arg5, %17 : tensor<128x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<128x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg4, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x50x768xf32>
    %27 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %28 = stablehlo.add %arg0, %27 : tensor<128x50x768xf32>
    %29 = stablehlo.add %26, %28 : tensor<128x50x768xf32>
    return %29 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.270(%arg0: tensor<768x768xf32>) -> tensor<128x768x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1 = mhlo.copy %0 : tensor<768x768xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    return %2 : tensor<128x768x768xf32>
  }
  func.func private @fused_computation.271(%arg0: tensor<1536x50x64xf32>) -> tensor<128x50x768xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
    %2 = mhlo.copy %1 : tensor<128x50x12x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
    return %3 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.272(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %4 = mhlo.copy %3 : tensor<128x12x50x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %5 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.273(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<128x12x50xf32>, %arg2: tensor<128x12x50xi1>) -> tensor<1536x50x50xf32> {
    %0 = stablehlo.not %arg2 : tensor<128x12x50xi1>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.divide %arg0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.select %1, %2, %4 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
    return %6 : tensor<1536x50x50xf32>
  }
  func.func private @fused_computation.274(%arg0: tensor<128x12x50xf32>, %arg1: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<128x12x50xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<128x12x50x50xf32>
    %5 = stablehlo.exponential %4 : tensor<128x12x50x50xf32>
    return %5 : tensor<128x12x50x50xf32>
  }
  func.func private @fused_computation.275(%arg0: tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
    %2 = stablehlo.compare EQ, %0, %1 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
    %3 = stablehlo.not %2 : tensor<128x12x50x50xi1>
    return %3 : tensor<128x12x50x50xi1>
  }
  func.func private @fused_computation.276(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x64x50xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
    %6 = mhlo.copy %5 : tensor<128x12x64x50xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
    return %7 : tensor<1536x64x50xf32>
  }
  func.func private @fused_computation.277(%arg0: tensor<6400x768xf32>, %arg1: tensor<768xf32>) -> tensor<1536x50x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<6400x768xf32>
    %cst = stablehlo.constant dense<0.353553385> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<6400x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
    %6 = mhlo.copy %5 : tensor<128x12x50x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
    return %7 : tensor<1536x50x64xf32>
  }
  func.func private @fused_computation.278(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<128x50xf32>, %arg7: tensor<128x50xf32>, %arg8: tensor<128x50x768xf32>, %arg9: tensor<50x1x768xf32>, %arg10: tensor<1x50xi32>) -> tensor<6400x768xf32> {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
    %16 = stablehlo.add %arg8, %15 : tensor<128x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %18 = stablehlo.multiply %arg7, %17 : tensor<128x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<128x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x50x768xf32>
    %27 = stablehlo.multiply %arg3, %17 : tensor<128x50xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %29 = stablehlo.subtract %26, %28 : tensor<128x50x768xf32>
    %30 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<128x50x768xf32>
    %32 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x50x768xf32>
    %34 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %35 = stablehlo.add %33, %34 : tensor<128x50x768xf32>
    %36 = mhlo.bitcast %35 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
    return %36 : tensor<6400x768xf32>
  }
  func.func private @fused_computation.279(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.280(%arg0: tensor<128x50xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<128x50xf32>, %arg4: tensor<128x50xf32>, %arg5: tensor<128x50x768xf32>, %arg6: tensor<50x1x768xf32>, %arg7: tensor<1x50xi32>) -> tensor<128x50x768xf32> {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
    %16 = stablehlo.add %arg5, %15 : tensor<128x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %18 = stablehlo.multiply %arg4, %17 : tensor<128x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<128x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x50x768xf32>
    %27 = stablehlo.multiply %arg0, %17 : tensor<128x50xf32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %29 = stablehlo.subtract %26, %28 : tensor<128x50x768xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<128x50x768xf32>
    return %30 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.281(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<128x50xf32>, %arg3: tensor<128x50xf32>, %arg4: tensor<128x50x768xf32>, %arg5: tensor<50x1x768xf32>, %arg6: tensor<1x50xi32>) -> tensor<128x50x768xf32> {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
    %16 = stablehlo.add %arg4, %15 : tensor<128x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %18 = stablehlo.multiply %arg3, %17 : tensor<128x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<128x50x768xf32>
    %21 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x50x768xf32>
    %23 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x50x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
    %26 = stablehlo.add %24, %25 : tensor<128x50x768xf32>
    return %26 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.282(%arg0: tensor<128x50xf32>) -> tensor<128x50xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x50xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %3 = stablehlo.add %1, %2 : tensor<128x50xf32>
    %4 = stablehlo.rsqrt %3 : tensor<128x50xf32>
    return %4 : tensor<128x50xf32>
  }
  func.func private @fused_computation.283(%arg0: tensor<128x50xf32>, %arg1: tensor<128x50x768xf32>, %arg2: tensor<50x1x768xf32>, %arg3: tensor<1x50xi32>) -> tensor<128x50x768xf32> {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
    %16 = stablehlo.add %arg1, %15 : tensor<128x50x768xf32>
    %cst_2 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
    %18 = stablehlo.multiply %arg0, %17 : tensor<128x50xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<128x50x768xf32>
    %21 = stablehlo.multiply %20, %20 : tensor<128x50x768xf32>
    return %21 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.284(%arg0: tensor<128x50x768xf32>, %arg1: tensor<50x1x768xf32>, %arg2: tensor<1x50xi32>) -> tensor<128x50x768xf32> {
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
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
    %16 = stablehlo.add %arg0, %15 : tensor<128x50x768xf32>
    return %16 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.285(%arg0: tensor<50x768xf32>, %arg1: tensor<1x50xi32>) -> tensor<50x1x768xf32> {
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
  func.func private @fused_computation.286(%arg0: tensor<128x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<128x50x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<768xf32>) -> tensor<128x1x768xf32>
    %1 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,768,7,7]{1,3,2,0}"} : (tensor<128x7x7x768xf32>) -> tensor<128x768x7x7xf32>
    %2 = mhlo.copy %1 : tensor<128x768x7x7xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x768x7x7xf32>) -> tensor<128x768x49xf32>
    %4 = stablehlo.transpose %3, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[128,49,768]{1,2,0}"} : (tensor<128x768x49xf32>) -> tensor<128x49x768xf32>
    %5 = mhlo.copy %4 : tensor<128x49x768xf32>
    %6 = stablehlo.concatenate %0, %5, dim = 1 : (tensor<128x1x768xf32>, tensor<128x49x768xf32>) -> tensor<128x50x768xf32>
    return %6 : tensor<128x50x768xf32>
  }
  func.func private @fused_computation.287(%arg0: tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[32,32,3,768]{1,0,2,3}"} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
    %1 = mhlo.copy %0 : tensor<32x32x3x768xf32>
    return %1 : tensor<32x32x3x768xf32>
  }
  func.func private @fused_computation.288(%arg0: tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,224,224,3]{2,1,3,0}"} : (tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<128x224x224x3xf32>
    return %1 : tensor<128x224x224x3xf32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_5.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_7.12(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_7.12.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_10.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_12.18(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_12.18.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_15.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_17.23(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_17.23.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_20.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_22.28(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_22.28.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_25.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_27.33(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_27.33.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_30.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_32.38(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_32.38.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_35.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_37.43(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_37.43.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_40.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_42.48(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_42.48.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.33(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_45.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.34(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_47.53(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.35(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_47.53.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.36(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.37(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_50.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.38(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_52.58(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.39(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_52.58.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.40(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.41(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_55.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.42(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_57.63(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.43(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_57.63.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.44(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_3.6.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.45(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_60.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.46(%arg0: tensor<128x12x50x50xf32>, %arg1: tensor<f32>) -> tensor<128x12x50x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    return %0 : tensor<128x12x50x2xf32>
  }
  func.func private @region_62.68(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_window_computation.47(%arg0: tensor<128x12x50x50xi1>, %arg1: tensor<i1>) -> tensor<128x12x50x2xi1> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %1 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    return %0 : tensor<128x12x50x2xi1>
  }
  func.func private @region_62.68.clone(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<128x12x50x2xi1>, %arg1: tensor<i1>) -> tensor<128x12x50xi1> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    return %0 : tensor<128x12x50xi1>
  }
  func.func private @region_3.6.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.48(%arg0: tensor<128x50x768xf32>, %arg1: tensor<f32>) -> tensor<128x50x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    return %0 : tensor<128x50x24xf32>
  }
  func.func private @region_64.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.49(%arg0: tensor<128x49x768xf32>, %arg1: tensor<f32>) -> tensor<128x2x768xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [7, 8], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 32, 1>, window_strides = array<i64: 1, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<128x49x768xf32>, tensor<f32>) -> tensor<128x2x768xf32>
    return %0 : tensor<128x2x768xf32>
  }
  func.func @main(%arg0: tensor<768xf32>, %arg1: tensor<768x3x32x32xf32>, %arg2: tensor<50x768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x768xf32>, %arg8: tensor<768xf32>, %arg9: tensor<768x768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768xf32>, %arg13: tensor<768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<3072x768xf32>, %arg16: tensor<3072xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x768xf32>, %arg28: tensor<768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<3072x768xf32>, %arg32: tensor<3072xf32>, %arg33: tensor<768x3072xf32>, %arg34: tensor<768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x768xf32>, %arg38: tensor<768xf32>, %arg39: tensor<768x768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<3072x768xf32>, %arg48: tensor<3072xf32>, %arg49: tensor<768x3072xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768xf32>, %arg52: tensor<768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768xf32>, %arg55: tensor<768x768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x768xf32>, %arg58: tensor<768xf32>, %arg59: tensor<768x768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768xf32>, %arg62: tensor<768xf32>, %arg63: tensor<3072x768xf32>, %arg64: tensor<3072xf32>, %arg65: tensor<768x3072xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768xf32>, %arg68: tensor<768xf32>, %arg69: tensor<768x768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768xf32>, %arg75: tensor<768x768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768xf32>, %arg78: tensor<768xf32>, %arg79: tensor<3072x768xf32>, %arg80: tensor<3072xf32>, %arg81: tensor<768x3072xf32>, %arg82: tensor<768xf32>, %arg83: tensor<768xf32>, %arg84: tensor<768xf32>, %arg85: tensor<768x768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x768xf32>, %arg88: tensor<768xf32>, %arg89: tensor<768x768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768xf32>, %arg93: tensor<768xf32>, %arg94: tensor<768xf32>, %arg95: tensor<3072x768xf32>, %arg96: tensor<3072xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768xf32>, %arg105: tensor<768x768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x768xf32>, %arg108: tensor<768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<3072x768xf32>, %arg112: tensor<3072xf32>, %arg113: tensor<768x3072xf32>, %arg114: tensor<768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x768xf32>, %arg118: tensor<768xf32>, %arg119: tensor<768x768xf32>, %arg120: tensor<768xf32>, %arg121: tensor<768x768xf32>, %arg122: tensor<768xf32>, %arg123: tensor<768x768xf32>, %arg124: tensor<768xf32>, %arg125: tensor<768xf32>, %arg126: tensor<768xf32>, %arg127: tensor<3072x768xf32>, %arg128: tensor<3072xf32>, %arg129: tensor<768x3072xf32>, %arg130: tensor<768xf32>, %arg131: tensor<768xf32>, %arg132: tensor<768xf32>, %arg133: tensor<768x768xf32>, %arg134: tensor<768xf32>, %arg135: tensor<768x768xf32>, %arg136: tensor<768xf32>, %arg137: tensor<768x768xf32>, %arg138: tensor<768xf32>, %arg139: tensor<768x768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768xf32>, %arg142: tensor<768xf32>, %arg143: tensor<3072x768xf32>, %arg144: tensor<3072xf32>, %arg145: tensor<768x3072xf32>, %arg146: tensor<768xf32>, %arg147: tensor<768xf32>, %arg148: tensor<768xf32>, %arg149: tensor<768x768xf32>, %arg150: tensor<768xf32>, %arg151: tensor<768x768xf32>, %arg152: tensor<768xf32>, %arg153: tensor<768x768xf32>, %arg154: tensor<768xf32>, %arg155: tensor<768x768xf32>, %arg156: tensor<768xf32>, %arg157: tensor<768xf32>, %arg158: tensor<768xf32>, %arg159: tensor<3072x768xf32>, %arg160: tensor<3072xf32>, %arg161: tensor<768x3072xf32>, %arg162: tensor<768xf32>, %arg163: tensor<768xf32>, %arg164: tensor<768xf32>, %arg165: tensor<768x768xf32>, %arg166: tensor<768xf32>, %arg167: tensor<768x768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768x768xf32>, %arg170: tensor<768xf32>, %arg171: tensor<768x768xf32>, %arg172: tensor<768xf32>, %arg173: tensor<768xf32>, %arg174: tensor<768xf32>, %arg175: tensor<3072x768xf32>, %arg176: tensor<3072xf32>, %arg177: tensor<768x3072xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<768xf32>, %arg181: tensor<768x768xf32>, %arg182: tensor<768xf32>, %arg183: tensor<768x768xf32>, %arg184: tensor<768xf32>, %arg185: tensor<768x768xf32>, %arg186: tensor<768xf32>, %arg187: tensor<768x768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<768xf32>, %arg191: tensor<3072x768xf32>, %arg192: tensor<3072xf32>, %arg193: tensor<768x3072xf32>, %arg194: tensor<768xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<2x768xf32>, %arg198: tensor<2xf32>, %arg199: tensor<1x50xi32>, %arg200: tensor<128x3x224x224xf32>) -> tensor<128x2xf32> {
    %0 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x3x224x224xf32>):
      %461 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,224,224,3]{2,1,3,0}"} : (tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32>
      %462 = mhlo.copy %461 : tensor<128x224x224x3xf32>
      mhlo.return %462 : tensor<128x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<128x3x224x224xf32>) -> tensor<128x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x3x32x32xf32>):
      %461 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[32,32,3,768]{1,0,2,3}"} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
      %462 = mhlo.copy %461 : tensor<32x32x3x768xf32>
      mhlo.return %462 : tensor<32x32x3x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x3x32x32xf32>) -> tensor<32x32x3x768xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [32, 32], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<128x224x224x3xf32>, tensor<32x32x3x768xf32>) -> tensor<128x7x7x768xf32>
    %3 = "mhlo.fusion"(%2, %arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x7x7x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x1x768xf32>
      %462 = stablehlo.transpose %arg201, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,768,7,7]{1,3,2,0}"} : (tensor<128x7x7x768xf32>) -> tensor<128x768x7x7xf32>
      %463 = mhlo.copy %462 : tensor<128x768x7x7xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x768x7x7xf32>) -> tensor<128x768x49xf32>
      %465 = stablehlo.transpose %464, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[128,49,768]{1,2,0}"} : (tensor<128x768x49xf32>) -> tensor<128x49x768xf32>
      %466 = mhlo.copy %465 : tensor<128x49x768xf32>
      %467 = stablehlo.concatenate %461, %466, dim = 1 : (tensor<128x1x768xf32>, tensor<128x49x768xf32>) -> tensor<128x50x768xf32>
      mhlo.return %467 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x7x7x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %4 = "mhlo.fusion"(%arg2, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<50x768xf32>, %arg202: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg202, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg202, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg202 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi32>) -> tensor<50x1xi32>
      %467 = "stablehlo.gather"(%arg201, %466) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 768>}> : (tensor<50x768xf32>, tensor<50x1xi32>) -> tensor<50x1x768xf32>
      mhlo.return %467 : tensor<50x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<50x768xf32>, tensor<1x50xi32>) -> tensor<50x1x768xf32>
    %5 = "mhlo.fusion"(%3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<50x1x768xf32>, %arg203: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg203, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg203, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg203 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg202 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
      %477 = stablehlo.add %arg201, %476 : tensor<128x50x768xf32>
      mhlo.return %477 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<128x50x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %7 = "mhlo.fusion"(%6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<50x1x768xf32>, %arg204: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg204, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg204, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg204 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg203 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
      %477 = stablehlo.add %arg202, %476 : tensor<128x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %479 = stablehlo.multiply %arg201, %478 : tensor<128x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<128x50x768xf32>
      %482 = stablehlo.multiply %481, %481 : tensor<128x50x768xf32>
      mhlo.return %482 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<128x50x768xf32>
    %8 = "mhlo.fusion"(%7, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %9 = "mhlo.fusion"(%8, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %10 = "mhlo.fusion"(%9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %11 = "mhlo.fusion"(%arg4, %arg3, %10, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50xf32>, %arg205: tensor<128x50x768xf32>, %arg206: tensor<50x1x768xf32>, %arg207: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg207, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg207, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg207 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg206 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
      %477 = stablehlo.add %arg205, %476 : tensor<128x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %479 = stablehlo.multiply %arg204, %478 : tensor<128x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<128x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<128x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<128x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<128x50x768xf32>
      mhlo.return %487 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<128x50x768xf32>
    %12 = "mhlo.fusion"(%11, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %13 = "mhlo.fusion"(%12, %arg4, %arg3, %10, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<128x50xf32>, %arg205: tensor<128x50xf32>, %arg206: tensor<128x50x768xf32>, %arg207: tensor<50x1x768xf32>, %arg208: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg208, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg208, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg208 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg207 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
      %477 = stablehlo.add %arg206, %476 : tensor<128x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %479 = stablehlo.multiply %arg205, %478 : tensor<128x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<128x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg204, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<128x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<128x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<128x50x768xf32>
      %488 = stablehlo.multiply %arg201, %478 : tensor<128x50xf32>
      %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %490 = stablehlo.subtract %487, %489 : tensor<128x50x768xf32>
      %491 = stablehlo.multiply %490, %490 : tensor<128x50x768xf32>
      mhlo.return %491 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<128x50x768xf32>
    %14 = "mhlo.fusion"(%13, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %15 = "mhlo.fusion"(%14, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %16 = "mhlo.fusion"(%15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %17 = "mhlo.fusion"(%arg14, %arg13, %16, %12, %arg4, %arg3, %10, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50xf32>, %arg205: tensor<768xf32>, %arg206: tensor<768xf32>, %arg207: tensor<128x50xf32>, %arg208: tensor<128x50xf32>, %arg209: tensor<128x50x768xf32>, %arg210: tensor<50x1x768xf32>, %arg211: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg211, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg211, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg211 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg210 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
      %477 = stablehlo.add %arg209, %476 : tensor<128x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %479 = stablehlo.multiply %arg208, %478 : tensor<128x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<128x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg207, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<128x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg206, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<128x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg205, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<128x50x768xf32>
      %488 = stablehlo.multiply %arg204, %478 : tensor<128x50xf32>
      %489 = stablehlo.broadcast_in_dim %488, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %490 = stablehlo.subtract %487, %489 : tensor<128x50x768xf32>
      %491 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %492 = stablehlo.multiply %490, %491 : tensor<128x50x768xf32>
      %493 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %494 = stablehlo.multiply %492, %493 : tensor<128x50x768xf32>
      %495 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %496 = stablehlo.add %494, %495 : tensor<128x50x768xf32>
      %497 = mhlo.bitcast %496 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %497 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50xf32>, tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<6400x768xf32>
    %18 = stablehlo.dot_general %17, %arg9, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %19 = "mhlo.fusion"(%18, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %20 = stablehlo.dot_general %17, %arg5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %21 = "mhlo.fusion"(%20, %arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %22 = stablehlo.dot_general %19, %21, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %24 = "mhlo.fusion"(%23, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %25 = "mhlo.fusion"(%24, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %26 = "mhlo.fusion"(%25, %22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %27 = "mhlo.fusion"(%26, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %28 = "mhlo.fusion"(%22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %29 = "mhlo.fusion"(%28, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %30 = "mhlo.fusion"(%29, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %31 = "mhlo.fusion"(%26, %27, %30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %32 = stablehlo.dot_general %17, %arg7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %33 = "mhlo.fusion"(%32, %arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %34 = stablehlo.dot_general %31, %33, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %35 = "mhlo.fusion"(%34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %36 = "mhlo.fusion"(%arg11) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %37 = stablehlo.dot_general %35, %36, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %38 = "mhlo.fusion"(%37, %arg12, %arg4, %arg3, %10, %6, %3, %4, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<768xf32>, %arg204: tensor<768xf32>, %arg205: tensor<128x50xf32>, %arg206: tensor<128x50xf32>, %arg207: tensor<128x50x768xf32>, %arg208: tensor<50x1x768xf32>, %arg209: tensor<1x50xi32>):
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %461 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %462 = stablehlo.compare LT, %arg209, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_2 = stablehlo.constant dense<50> : tensor<i32>
      %463 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %464 = stablehlo.add %arg209, %463 : tensor<1x50xi32>
      %465 = stablehlo.select %462, %464, %arg209 : tensor<1x50xi1>, tensor<1x50xi32>
      %466 = stablehlo.compare GE, %465, %461 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %c_3 = stablehlo.constant dense<49> : tensor<i32>
      %467 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x50xi32>
      %468 = stablehlo.compare LE, %465, %467 : (tensor<1x50xi32>, tensor<1x50xi32>) -> tensor<1x50xi1>
      %469 = stablehlo.and %466, %468 : tensor<1x50xi1>
      %470 = mhlo.bitcast %469 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1x50xi1>) -> tensor<50xi1>
      %471 = stablehlo.broadcast_in_dim %470, dims = [1] : (tensor<50xi1>) -> tensor<1x50x768xi1>
      %472 = mhlo.bitcast %arg208 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<50x1x768xf32>) -> tensor<1x50x768xf32>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x50x768xf32>
      %474 = stablehlo.select %471, %472, %473 : tensor<1x50x768xi1>, tensor<1x50x768xf32>
      %475 = mhlo.bitcast %474 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x50x768xf32>) -> tensor<50x768xf32>
      %476 = stablehlo.broadcast_in_dim %475, dims = [1, 2] : (tensor<50x768xf32>) -> tensor<128x50x768xf32>
      %477 = stablehlo.add %arg207, %476 : tensor<128x50x768xf32>
      %cst_5 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %479 = stablehlo.multiply %arg206, %478 : tensor<128x50xf32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %481 = stablehlo.subtract %477, %480 : tensor<128x50x768xf32>
      %482 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %483 = stablehlo.multiply %481, %482 : tensor<128x50x768xf32>
      %484 = stablehlo.broadcast_in_dim %arg204, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %485 = stablehlo.multiply %483, %484 : tensor<128x50x768xf32>
      %486 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %487 = stablehlo.add %485, %486 : tensor<128x50x768xf32>
      %488 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %489 = stablehlo.add %arg201, %488 : tensor<128x50x768xf32>
      %490 = stablehlo.add %487, %489 : tensor<128x50x768xf32>
      mhlo.return %490 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<50x1x768xf32>, tensor<1x50xi32>) -> tensor<128x50x768xf32>
    %39 = "mhlo.fusion"(%38, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %40 = "mhlo.fusion"(%38, %39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %41 = "mhlo.fusion"(%40, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %42 = "mhlo.fusion"(%41, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %43 = "mhlo.fusion"(%42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %44 = "mhlo.fusion"(%arg20, %arg19, %43, %38, %39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %45 = stablehlo.dot_general %44, %arg15, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %46 = "mhlo.fusion"(%45, %arg16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %47 = stablehlo.dot_general %46, %arg17, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %48 = "mhlo.fusion"(%38, %47, %arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %49 = "mhlo.fusion"(%48, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %50 = "mhlo.fusion"(%48, %49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %51 = "mhlo.fusion"(%50, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %52 = "mhlo.fusion"(%51, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %53 = "mhlo.fusion"(%52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %54 = "mhlo.fusion"(%arg30, %arg29, %53, %48, %49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %55 = stablehlo.dot_general %54, %arg25, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %56 = "mhlo.fusion"(%55, %arg26) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %57 = stablehlo.dot_general %54, %arg21, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %58 = "mhlo.fusion"(%57, %arg22) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %59 = stablehlo.dot_general %56, %58, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %60 = mhlo.bitcast %59 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %61 = "mhlo.fusion"(%60, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %62 = "mhlo.fusion"(%61, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %63 = "mhlo.fusion"(%62, %59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %64 = "mhlo.fusion"(%63, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %65 = "mhlo.fusion"(%59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %66 = "mhlo.fusion"(%65, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %67 = "mhlo.fusion"(%66, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %68 = "mhlo.fusion"(%63, %64, %67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %69 = stablehlo.dot_general %54, %arg23, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %70 = "mhlo.fusion"(%69, %arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %71 = stablehlo.dot_general %68, %70, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %72 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %73 = "mhlo.fusion"(%arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %74 = stablehlo.dot_general %72, %73, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %75 = "mhlo.fusion"(%48, %74, %arg28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %76 = "mhlo.fusion"(%75, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %77 = "mhlo.fusion"(%75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %78 = "mhlo.fusion"(%77, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %79 = "mhlo.fusion"(%78, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %80 = "mhlo.fusion"(%79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %81 = "mhlo.fusion"(%arg36, %arg35, %80, %75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %82 = stablehlo.dot_general %81, %arg31, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %83 = "mhlo.fusion"(%82, %arg32) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %84 = stablehlo.dot_general %83, %arg33, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %85 = "mhlo.fusion"(%75, %84, %arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %86 = "mhlo.fusion"(%85, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %87 = "mhlo.fusion"(%85, %86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %88 = "mhlo.fusion"(%87, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %89 = "mhlo.fusion"(%88, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %91 = "mhlo.fusion"(%arg46, %arg45, %90, %85, %86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %92 = stablehlo.dot_general %91, %arg41, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %93 = "mhlo.fusion"(%92, %arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %94 = stablehlo.dot_general %91, %arg37, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %95 = "mhlo.fusion"(%94, %arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %96 = stablehlo.dot_general %93, %95, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %98 = "mhlo.fusion"(%97, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %99 = "mhlo.fusion"(%98, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %100 = "mhlo.fusion"(%99, %96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %101 = "mhlo.fusion"(%100, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %102 = "mhlo.fusion"(%96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %103 = "mhlo.fusion"(%102, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %104 = "mhlo.fusion"(%103, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %105 = "mhlo.fusion"(%100, %101, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %106 = stablehlo.dot_general %91, %arg39, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %107 = "mhlo.fusion"(%106, %arg40) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %108 = stablehlo.dot_general %105, %107, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %110 = "mhlo.fusion"(%arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %111 = stablehlo.dot_general %109, %110, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %112 = "mhlo.fusion"(%85, %111, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %113 = "mhlo.fusion"(%112, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %114 = "mhlo.fusion"(%112, %113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %115 = "mhlo.fusion"(%114, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %116 = "mhlo.fusion"(%115, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %117 = "mhlo.fusion"(%116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %118 = "mhlo.fusion"(%arg52, %arg51, %117, %112, %113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %119 = stablehlo.dot_general %118, %arg47, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %120 = "mhlo.fusion"(%119, %arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %121 = stablehlo.dot_general %120, %arg49, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %122 = "mhlo.fusion"(%112, %121, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %123 = "mhlo.fusion"(%122, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %124 = "mhlo.fusion"(%122, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %125 = "mhlo.fusion"(%124, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %126 = "mhlo.fusion"(%125, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %128 = "mhlo.fusion"(%arg62, %arg61, %127, %122, %123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %129 = stablehlo.dot_general %128, %arg57, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %130 = "mhlo.fusion"(%129, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %131 = stablehlo.dot_general %128, %arg53, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %132 = "mhlo.fusion"(%131, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %133 = stablehlo.dot_general %130, %132, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %135 = "mhlo.fusion"(%134, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %136 = "mhlo.fusion"(%135, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %137 = "mhlo.fusion"(%136, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %138 = "mhlo.fusion"(%137, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %139 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %140 = "mhlo.fusion"(%139, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %141 = "mhlo.fusion"(%140, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %142 = "mhlo.fusion"(%137, %138, %141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %143 = stablehlo.dot_general %128, %arg55, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %144 = "mhlo.fusion"(%143, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %145 = stablehlo.dot_general %142, %144, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %146 = "mhlo.fusion"(%145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %147 = "mhlo.fusion"(%arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %148 = stablehlo.dot_general %146, %147, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %149 = "mhlo.fusion"(%122, %148, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %150 = "mhlo.fusion"(%149, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %151 = "mhlo.fusion"(%149, %150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %152 = "mhlo.fusion"(%151, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %153 = "mhlo.fusion"(%152, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %155 = "mhlo.fusion"(%arg68, %arg67, %154, %149, %150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %156 = stablehlo.dot_general %155, %arg63, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %157 = "mhlo.fusion"(%156, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %158 = stablehlo.dot_general %157, %arg65, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %159 = "mhlo.fusion"(%149, %158, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %160 = "mhlo.fusion"(%159, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %161 = "mhlo.fusion"(%159, %160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %162 = "mhlo.fusion"(%161, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %163 = "mhlo.fusion"(%162, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %165 = "mhlo.fusion"(%arg78, %arg77, %164, %159, %160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %166 = stablehlo.dot_general %165, %arg73, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %167 = "mhlo.fusion"(%166, %arg74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %168 = stablehlo.dot_general %165, %arg69, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %169 = "mhlo.fusion"(%168, %arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %170 = stablehlo.dot_general %167, %169, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %172 = "mhlo.fusion"(%171, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %173 = "mhlo.fusion"(%172, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %174 = "mhlo.fusion"(%173, %170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %175 = "mhlo.fusion"(%174, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %176 = "mhlo.fusion"(%170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %177 = "mhlo.fusion"(%176, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %178 = "mhlo.fusion"(%177, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %179 = "mhlo.fusion"(%174, %175, %178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %180 = stablehlo.dot_general %165, %arg71, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %181 = "mhlo.fusion"(%180, %arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %182 = stablehlo.dot_general %179, %181, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %184 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %185 = stablehlo.dot_general %183, %184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %186 = "mhlo.fusion"(%159, %185, %arg76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %187 = "mhlo.fusion"(%186, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %188 = "mhlo.fusion"(%186, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %189 = "mhlo.fusion"(%188, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %190 = "mhlo.fusion"(%189, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %192 = "mhlo.fusion"(%arg84, %arg83, %191, %186, %187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %193 = stablehlo.dot_general %192, %arg79, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %194 = "mhlo.fusion"(%193, %arg80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %195 = stablehlo.dot_general %194, %arg81, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %196 = "mhlo.fusion"(%186, %195, %arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %197 = "mhlo.fusion"(%196, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %198 = "mhlo.fusion"(%196, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %199 = "mhlo.fusion"(%198, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %200 = "mhlo.fusion"(%199, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %202 = "mhlo.fusion"(%arg94, %arg93, %201, %196, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %203 = stablehlo.dot_general %202, %arg89, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %204 = "mhlo.fusion"(%203, %arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %205 = stablehlo.dot_general %202, %arg85, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %206 = "mhlo.fusion"(%205, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %207 = stablehlo.dot_general %204, %206, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %208 = mhlo.bitcast %207 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %209 = "mhlo.fusion"(%208, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %210 = "mhlo.fusion"(%209, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %211 = "mhlo.fusion"(%210, %207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %212 = "mhlo.fusion"(%211, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %213 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %214 = "mhlo.fusion"(%213, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %215 = "mhlo.fusion"(%214, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %216 = "mhlo.fusion"(%211, %212, %215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %217 = stablehlo.dot_general %202, %arg87, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %218 = "mhlo.fusion"(%217, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %219 = stablehlo.dot_general %216, %218, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %221 = "mhlo.fusion"(%arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %222 = stablehlo.dot_general %220, %221, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %223 = "mhlo.fusion"(%196, %222, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %224 = "mhlo.fusion"(%223, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %225 = "mhlo.fusion"(%223, %224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %226 = "mhlo.fusion"(%225, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %227 = "mhlo.fusion"(%226, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %228 = "mhlo.fusion"(%227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %229 = "mhlo.fusion"(%arg100, %arg99, %228, %223, %224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %230 = stablehlo.dot_general %229, %arg95, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %231 = "mhlo.fusion"(%230, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %232 = stablehlo.dot_general %231, %arg97, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %233 = "mhlo.fusion"(%223, %232, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %234 = "mhlo.fusion"(%233, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %235 = "mhlo.fusion"(%233, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %236 = "mhlo.fusion"(%235, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %237 = "mhlo.fusion"(%236, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %238 = "mhlo.fusion"(%237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %239 = "mhlo.fusion"(%arg110, %arg109, %238, %233, %234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %240 = stablehlo.dot_general %239, %arg105, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %241 = "mhlo.fusion"(%240, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %242 = stablehlo.dot_general %239, %arg101, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %243 = "mhlo.fusion"(%242, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %244 = stablehlo.dot_general %241, %243, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %245 = mhlo.bitcast %244 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %246 = "mhlo.fusion"(%245, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %247 = "mhlo.fusion"(%246, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %248 = "mhlo.fusion"(%247, %244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %249 = "mhlo.fusion"(%248, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %250 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %251 = "mhlo.fusion"(%250, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %252 = "mhlo.fusion"(%251, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %253 = "mhlo.fusion"(%248, %249, %252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %254 = stablehlo.dot_general %239, %arg103, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %255 = "mhlo.fusion"(%254, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %256 = stablehlo.dot_general %253, %255, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %257 = "mhlo.fusion"(%256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %258 = "mhlo.fusion"(%arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %259 = stablehlo.dot_general %257, %258, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %260 = "mhlo.fusion"(%233, %259, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %261 = "mhlo.fusion"(%260, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %262 = "mhlo.fusion"(%260, %261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %263 = "mhlo.fusion"(%262, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %264 = "mhlo.fusion"(%263, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %265 = "mhlo.fusion"(%264) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %266 = "mhlo.fusion"(%arg116, %arg115, %265, %260, %261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %267 = stablehlo.dot_general %266, %arg111, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %268 = "mhlo.fusion"(%267, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %269 = stablehlo.dot_general %268, %arg113, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %270 = "mhlo.fusion"(%260, %269, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %271 = "mhlo.fusion"(%270, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %272 = "mhlo.fusion"(%270, %271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %273 = "mhlo.fusion"(%272, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %274 = "mhlo.fusion"(%273, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %275 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %276 = "mhlo.fusion"(%arg126, %arg125, %275, %270, %271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %277 = stablehlo.dot_general %276, %arg121, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %278 = "mhlo.fusion"(%277, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %279 = stablehlo.dot_general %276, %arg117, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %280 = "mhlo.fusion"(%279, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %281 = stablehlo.dot_general %278, %280, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %283 = "mhlo.fusion"(%282, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %284 = "mhlo.fusion"(%283, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %285 = "mhlo.fusion"(%284, %281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %286 = "mhlo.fusion"(%285, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %287 = "mhlo.fusion"(%281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %288 = "mhlo.fusion"(%287, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %289 = "mhlo.fusion"(%288, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %290 = "mhlo.fusion"(%285, %286, %289) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %291 = stablehlo.dot_general %276, %arg119, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %292 = "mhlo.fusion"(%291, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %293 = stablehlo.dot_general %290, %292, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %294 = "mhlo.fusion"(%293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %295 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %296 = stablehlo.dot_general %294, %295, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %297 = "mhlo.fusion"(%270, %296, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %298 = "mhlo.fusion"(%297, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %299 = "mhlo.fusion"(%297, %298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %300 = "mhlo.fusion"(%299, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %301 = "mhlo.fusion"(%300, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %302 = "mhlo.fusion"(%301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %303 = "mhlo.fusion"(%arg132, %arg131, %302, %297, %298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %304 = stablehlo.dot_general %303, %arg127, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %305 = "mhlo.fusion"(%304, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %306 = stablehlo.dot_general %305, %arg129, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %307 = "mhlo.fusion"(%297, %306, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %308 = "mhlo.fusion"(%307, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %309 = "mhlo.fusion"(%307, %308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %310 = "mhlo.fusion"(%309, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %311 = "mhlo.fusion"(%310, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %312 = "mhlo.fusion"(%311) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %313 = "mhlo.fusion"(%arg142, %arg141, %312, %307, %308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %314 = stablehlo.dot_general %313, %arg137, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %315 = "mhlo.fusion"(%314, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %316 = stablehlo.dot_general %313, %arg133, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %317 = "mhlo.fusion"(%316, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %318 = stablehlo.dot_general %315, %317, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %320 = "mhlo.fusion"(%319, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %322 = "mhlo.fusion"(%321, %318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %323 = "mhlo.fusion"(%322, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %324 = "mhlo.fusion"(%318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %325 = "mhlo.fusion"(%324, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %326 = "mhlo.fusion"(%325, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %327 = "mhlo.fusion"(%322, %323, %326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %328 = stablehlo.dot_general %313, %arg135, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %329 = "mhlo.fusion"(%328, %arg136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %330 = stablehlo.dot_general %327, %329, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %331 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %332 = "mhlo.fusion"(%arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %333 = stablehlo.dot_general %331, %332, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %334 = "mhlo.fusion"(%307, %333, %arg140) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %335 = "mhlo.fusion"(%334, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %336 = "mhlo.fusion"(%334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %338 = "mhlo.fusion"(%337, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %339 = "mhlo.fusion"(%338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %340 = "mhlo.fusion"(%arg148, %arg147, %339, %334, %335) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %341 = stablehlo.dot_general %340, %arg143, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %342 = "mhlo.fusion"(%341, %arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %343 = stablehlo.dot_general %342, %arg145, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %344 = "mhlo.fusion"(%334, %343, %arg146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %345 = "mhlo.fusion"(%344, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %346 = "mhlo.fusion"(%344, %345) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %347 = "mhlo.fusion"(%346, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %348 = "mhlo.fusion"(%347, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %349 = "mhlo.fusion"(%348) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %350 = "mhlo.fusion"(%arg158, %arg157, %349, %344, %345) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %351 = stablehlo.dot_general %350, %arg153, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %352 = "mhlo.fusion"(%351, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %353 = stablehlo.dot_general %350, %arg149, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %354 = "mhlo.fusion"(%353, %arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %355 = stablehlo.dot_general %352, %354, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %356 = mhlo.bitcast %355 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %357 = "mhlo.fusion"(%356, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %358 = "mhlo.fusion"(%357, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %359 = "mhlo.fusion"(%358, %355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %360 = "mhlo.fusion"(%359, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %361 = "mhlo.fusion"(%355) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %362 = "mhlo.fusion"(%361, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %363 = "mhlo.fusion"(%362, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %364 = "mhlo.fusion"(%359, %360, %363) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %365 = stablehlo.dot_general %350, %arg151, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %366 = "mhlo.fusion"(%365, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %367 = stablehlo.dot_general %364, %366, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %368 = "mhlo.fusion"(%367) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %369 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %370 = stablehlo.dot_general %368, %369, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %371 = "mhlo.fusion"(%344, %370, %arg156) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %372 = "mhlo.fusion"(%371, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %373 = "mhlo.fusion"(%371, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %374 = "mhlo.fusion"(%373, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %375 = "mhlo.fusion"(%374, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %376 = "mhlo.fusion"(%375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %377 = "mhlo.fusion"(%arg164, %arg163, %376, %371, %372) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %378 = stablehlo.dot_general %377, %arg159, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %379 = "mhlo.fusion"(%378, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %380 = stablehlo.dot_general %379, %arg161, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %381 = "mhlo.fusion"(%371, %380, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %382 = "mhlo.fusion"(%381, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %383 = "mhlo.fusion"(%381, %382) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %384 = "mhlo.fusion"(%383, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %385 = "mhlo.fusion"(%384, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %386 = "mhlo.fusion"(%385) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %387 = "mhlo.fusion"(%arg174, %arg173, %386, %381, %382) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %388 = stablehlo.dot_general %387, %arg169, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %389 = "mhlo.fusion"(%388, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %390 = stablehlo.dot_general %387, %arg165, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %391 = "mhlo.fusion"(%390, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %392 = stablehlo.dot_general %389, %391, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %393 = mhlo.bitcast %392 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %394 = "mhlo.fusion"(%393, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %395 = "mhlo.fusion"(%394, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %396 = "mhlo.fusion"(%395, %392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %397 = "mhlo.fusion"(%396, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %398 = "mhlo.fusion"(%392) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %399 = "mhlo.fusion"(%398, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %400 = "mhlo.fusion"(%399, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %401 = "mhlo.fusion"(%396, %397, %400) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %402 = stablehlo.dot_general %387, %arg167, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %403 = "mhlo.fusion"(%402, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %404 = stablehlo.dot_general %401, %403, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %405 = "mhlo.fusion"(%404) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %406 = "mhlo.fusion"(%arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %407 = stablehlo.dot_general %405, %406, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %408 = "mhlo.fusion"(%381, %407, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %409 = "mhlo.fusion"(%408, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %410 = "mhlo.fusion"(%408, %409) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %411 = "mhlo.fusion"(%410, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %412 = "mhlo.fusion"(%411, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %413 = "mhlo.fusion"(%412) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %414 = "mhlo.fusion"(%arg180, %arg179, %413, %408, %409) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %415 = stablehlo.dot_general %414, %arg175, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %416 = "mhlo.fusion"(%415, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %417 = stablehlo.dot_general %416, %arg177, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %418 = "mhlo.fusion"(%408, %417, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %419 = "mhlo.fusion"(%418, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %420 = "mhlo.fusion"(%418, %419) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %421 = "mhlo.fusion"(%420, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %422 = "mhlo.fusion"(%421, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %423 = "mhlo.fusion"(%422) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %424 = "mhlo.fusion"(%arg190, %arg189, %423, %418, %419) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %425 = stablehlo.dot_general %424, %arg185, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %426 = "mhlo.fusion"(%425, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %467 = mhlo.copy %466 : tensor<128x12x50x64xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %468 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %427 = stablehlo.dot_general %424, %arg181, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %428 = "mhlo.fusion"(%427, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %cst_1 = stablehlo.constant dense<0.353553385> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6400x768xf32>
      %464 = stablehlo.multiply %462, %463 : tensor<6400x768xf32>
      %465 = mhlo.bitcast %464 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %466 = stablehlo.transpose %465, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,64,50]{2,1,3,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x64x50xf32>
      %467 = mhlo.copy %466 : tensor<128x12x64x50xf32>
      %468 = mhlo.bitcast %467 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x64x50xf32>) -> tensor<1536x64x50xf32>
      mhlo.return %468 : tensor<1536x64x50xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x64x50xf32>
    %429 = stablehlo.dot_general %426, %428, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x64xf32>, tensor<1536x64x50xf32>) -> tensor<1536x50x50xf32>
    %430 = mhlo.bitcast %429 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %431 = "mhlo.fusion"(%430, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.maximum %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
      mhlo.return %461 : tensor<128x12x50x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50x2xf32>
    %432 = "mhlo.fusion"(%431, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.maximum across dimensions = [3] : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %433 = "mhlo.fusion"(%432, %429) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50xf32>, %arg202: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50xf32>
      %463 = stablehlo.maximum %462, %arg201 : tensor<128x12x50xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.subtract %461, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.exponential %465 : tensor<128x12x50x50xf32>
      mhlo.return %466 : tensor<128x12x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50xf32>, tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
    %434 = "mhlo.fusion"(%433, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
      mhlo.return %461 : tensor<128x12x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<f32>) -> tensor<128x12x50xf32>
    %435 = "mhlo.fusion"(%429) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x50xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %463 = stablehlo.compare EQ, %461, %462 : (tensor<128x12x50x50xf32>, tensor<128x12x50x50xf32>) -> tensor<128x12x50x50xi1>
      %464 = stablehlo.not %463 : tensor<128x12x50x50xi1>
      mhlo.return %464 : tensor<128x12x50x50xi1>
    }) {output_operand_aliasing = []} : (tensor<1536x50x50xf32>) -> tensor<128x12x50x50xi1>
    %436 = "mhlo.fusion"(%435, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xi1>, %arg202: tensor<i1>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [7, 7]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<i1>, %arg204: tensor<i1>):
        %462 = stablehlo.or %arg203, %arg204 : tensor<i1>
        stablehlo.return %462 : tensor<i1>
      }) : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
      mhlo.return %461 : tensor<128x12x50x2xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xi1>, tensor<i1>) -> tensor<128x12x50x2xi1>
    %437 = "mhlo.fusion"(%436, %c) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x2xi1>, %arg202: tensor<i1>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.or across dimensions = [3] : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
      mhlo.return %461 : tensor<128x12x50xi1>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x2xi1>, tensor<i1>) -> tensor<128x12x50xi1>
    %438 = "mhlo.fusion"(%433, %434, %437) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x12x50x50xf32>, %arg202: tensor<128x12x50xf32>, %arg203: tensor<128x12x50xi1>):
      %461 = stablehlo.not %arg203 : tensor<128x12x50xi1>
      %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<128x12x50xi1>) -> tensor<128x12x50x50xi1>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x12x50x50xf32>
      %464 = stablehlo.broadcast_in_dim %arg202, dims = [0, 1, 2] : (tensor<128x12x50xf32>) -> tensor<128x12x50x50xf32>
      %465 = stablehlo.divide %arg201, %464 : tensor<128x12x50x50xf32>
      %466 = stablehlo.select %462, %463, %465 : tensor<128x12x50x50xi1>, tensor<128x12x50x50xf32>
      %467 = mhlo.bitcast %466 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x50xf32>) -> tensor<1536x50x50xf32>
      mhlo.return %467 : tensor<1536x50x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x12x50x50xf32>, tensor<128x12x50xf32>, tensor<128x12x50xi1>) -> tensor<1536x50x50xf32>
    %439 = stablehlo.dot_general %424, %arg183, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<768x768xf32>) -> tensor<6400x768xf32>
    %440 = "mhlo.fusion"(%439, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x768xf32>, %arg202: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x12x64xf32>
      %464 = stablehlo.transpose %463, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,12,50,64]{3,1,2,0}"} : (tensor<128x50x12x64xf32>) -> tensor<128x12x50x64xf32>
      %465 = mhlo.copy %464 : tensor<128x12x50x64xf32>
      %466 = mhlo.bitcast %465 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x12x50x64xf32>) -> tensor<1536x50x64xf32>
      mhlo.return %466 : tensor<1536x50x64xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x768xf32>, tensor<768xf32>) -> tensor<1536x50x64xf32>
    %441 = stablehlo.dot_general %438, %440, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1536x50x50xf32>, tensor<1536x50x64xf32>) -> tensor<1536x50x64xf32>
    %442 = "mhlo.fusion"(%441) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1536x50x64xf32>):
      %461 = mhlo.bitcast %arg201 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1536x50x64xf32>) -> tensor<128x12x50x64xf32>
      %462 = stablehlo.transpose %461, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[128,50,12,64]{3,1,2,0}"} : (tensor<128x12x50x64xf32>) -> tensor<128x50x12x64xf32>
      %463 = mhlo.copy %462 : tensor<128x50x12x64xf32>
      %464 = mhlo.bitcast %463 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<128x50x12x64xf32>) -> tensor<128x50x768xf32>
      mhlo.return %464 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<1536x50x64xf32>) -> tensor<128x50x768xf32>
    %443 = "mhlo.fusion"(%arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x768xf32>):
      %461 = stablehlo.transpose %arg201, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,768]{0,1}"} : (tensor<768x768xf32>) -> tensor<768x768xf32>
      %462 = mhlo.copy %461 : tensor<768x768xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [1, 2] : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
      mhlo.return %463 : tensor<128x768x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x768xf32>) -> tensor<128x768x768xf32>
    %444 = stablehlo.dot_general %442, %443, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x50x768xf32>, tensor<128x768x768xf32>) -> tensor<128x50x768xf32>
    %445 = "mhlo.fusion"(%418, %444, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %462 = stablehlo.add %arg202, %461 : tensor<128x50x768xf32>
      %463 = stablehlo.add %arg201, %462 : tensor<128x50x768xf32>
      mhlo.return %463 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50x768xf32>, tensor<768xf32>) -> tensor<128x50x768xf32>
    %446 = "mhlo.fusion"(%445, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50xf32>
    %447 = "mhlo.fusion"(%445, %446) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg202, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.multiply %464, %464 : tensor<128x50x768xf32>
      mhlo.return %465 : tensor<128x50x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<128x50x768xf32>
    %448 = "mhlo.fusion"(%447, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
      mhlo.return %461 : tensor<128x50x24xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<f32>) -> tensor<128x50x24xf32>
    %449 = "mhlo.fusion"(%448, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x50x24xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [2] : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
      mhlo.return %461 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x24xf32>, tensor<f32>) -> tensor<128x50xf32>
    %450 = "mhlo.fusion"(%449) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x50xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %464 = stablehlo.add %462, %463 : tensor<128x50xf32>
      %465 = stablehlo.rsqrt %464 : tensor<128x50xf32>
      mhlo.return %465 : tensor<128x50xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50xf32>) -> tensor<128x50xf32>
    %451 = "mhlo.fusion"(%arg196, %arg195, %450, %445, %446) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<128x50xf32>, %arg204: tensor<128x50x768xf32>, %arg205: tensor<128x50xf32>):
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50xf32>
      %462 = stablehlo.multiply %arg205, %461 : tensor<128x50xf32>
      %463 = stablehlo.broadcast_in_dim %462, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.subtract %arg204, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1] : (tensor<128x50xf32>) -> tensor<128x50x768xf32>
      %466 = stablehlo.multiply %464, %465 : tensor<128x50x768xf32>
      %467 = stablehlo.broadcast_in_dim %arg202, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %468 = stablehlo.multiply %466, %467 : tensor<128x50x768xf32>
      %469 = stablehlo.broadcast_in_dim %arg201, dims = [2] : (tensor<768xf32>) -> tensor<128x50x768xf32>
      %470 = stablehlo.add %468, %469 : tensor<128x50x768xf32>
      %471 = mhlo.bitcast %470 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x768xf32>) -> tensor<6400x768xf32>
      mhlo.return %471 : tensor<6400x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<128x50xf32>, tensor<128x50x768xf32>, tensor<128x50xf32>) -> tensor<6400x768xf32>
    %452 = stablehlo.dot_general %451, %arg191, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x768xf32>, tensor<3072x768xf32>) -> tensor<6400x3072xf32>
    %453 = "mhlo.fusion"(%452, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<6400x3072xf32>, %arg202: tensor<3072xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<6400x3072xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<6400x3072xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x3072xf32>) -> tensor<128x50x3072xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %cst_2 = stablehlo.constant dense<1.702000e+00> : tensor<f32>
      %465 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x50x3072xf32>
      %466 = stablehlo.multiply %463, %465 : tensor<128x50x3072xf32>
      %467 = stablehlo.negate %466 : tensor<128x50x3072xf32>
      %468 = stablehlo.exponential %467 : tensor<128x50x3072xf32>
      %469 = stablehlo.add %464, %468 : tensor<128x50x3072xf32>
      %470 = stablehlo.divide %464, %469 : tensor<128x50x3072xf32>
      %471 = stablehlo.multiply %463, %470 : tensor<128x50x3072xf32>
      %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x50x3072xf32>) -> tensor<6400x3072xf32>
      mhlo.return %472 : tensor<6400x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<6400x3072xf32>, tensor<3072xf32>) -> tensor<6400x3072xf32>
    %454 = stablehlo.dot_general %453, %arg193, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<6400x3072xf32>, tensor<768x3072xf32>) -> tensor<6400x768xf32>
    %455 = "mhlo.fusion"(%445, %454, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x50x768xf32>, %arg202: tensor<6400x768xf32>, %arg203: tensor<768xf32>):
      %461 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<6400x768xf32>
      %462 = stablehlo.add %461, %arg202 : tensor<6400x768xf32>
      %463 = mhlo.bitcast %462 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<6400x768xf32>) -> tensor<128x50x768xf32>
      %464 = stablehlo.add %arg201, %463 : tensor<128x50x768xf32>
      %465 = stablehlo.slice %464 [0:128, 1:50, 0:768] : (tensor<128x50x768xf32>) -> tensor<128x49x768xf32>
      mhlo.return %465 : tensor<128x49x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x50x768xf32>, tensor<6400x768xf32>, tensor<768xf32>) -> tensor<128x49x768xf32>
    %456 = "mhlo.fusion"(%455, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x49x768xf32>, %arg202: tensor<f32>):
      %461 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [7, 8], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 32, 1>, window_strides = array<i64: 1, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %462 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %462 : tensor<f32>
      }) : (tensor<128x49x768xf32>, tensor<f32>) -> tensor<128x2x768xf32>
      mhlo.return %461 : tensor<128x2x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x49x768xf32>, tensor<f32>) -> tensor<128x2x768xf32>
    %457 = "mhlo.fusion"(%456, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<128x2x768xf32>, %arg202: tensor<f32>):
      %461 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1] : (tensor<128x2x768xf32>, tensor<f32>) -> tensor<128x768xf32>
      mhlo.return %461 : tensor<128x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x2x768xf32>, tensor<f32>) -> tensor<128x768xf32>
    %458 = "mhlo.fusion"(%457) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x768xf32>):
      %cst_1 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x768xf32>
      %462 = stablehlo.multiply %arg201, %461 : tensor<128x768xf32>
      mhlo.return %462 : tensor<128x768xf32>
    }) {output_operand_aliasing = []} : (tensor<128x768xf32>) -> tensor<128x768xf32>
    %459 = stablehlo.dot_general %458, %arg197, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf32>, tensor<2x768xf32>) -> tensor<128x2xf32>
    %460 = "mhlo.fusion"(%459, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<128x2xf32>, %arg202: tensor<2xf32>):
      %461 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<2xf32>) -> tensor<128x2xf32>
      %462 = stablehlo.add %461, %arg201 : tensor<128x2xf32>
      mhlo.return %462 : tensor<128x2xf32>
    }) {output_operand_aliasing = []} : (tensor<128x2xf32>, tensor<2xf32>) -> tensor<128x2xf32>
    return %460 : tensor<128x2xf32>
  }
}
