module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_50.61.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<64x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_59.71(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_50.61.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<64x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_58.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_57.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<64x8x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x8x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    return %0 : tensor<64x8x256xf32>
  }
  func.func private @region_56.68.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<64x8x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x8x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x256x8xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    return %0 : tensor<64x8x256xf32>
  }
  func.func private @region_50.61.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<64x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_55.67(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_50.61.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<64x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_54.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_53.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<64x8x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x8x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    return %0 : tensor<64x8x256xf32>
  }
  func.func private @region_52.64.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<64x8x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x8x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x256x8xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    return %0 : tensor<64x8x256xf32>
  }
  func.func private @region_50.61.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<64x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_51.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_50.61.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<64x256x8xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_48.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_34.42.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<64x1024x5xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_47.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_34.42.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<64x1024x5xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_46.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_45.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<64x5x1024x256xf32>, %arg1: tensor<f32>) -> tensor<64x5x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    return %0 : tensor<64x5x1024xf32>
  }
  func.func private @region_44.55.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<64x5x1024x8xf32>, %arg1: tensor<f32>) -> tensor<64x5x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<64x5x1024x8xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    return %0 : tensor<64x5x1024xf32>
  }
  func.func private @region_38.48.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<64x256x5xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x5xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_43.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.23(%arg0: tensor<64x256x160xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_34.42.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.24(%arg0: tensor<64x1024x5xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_42.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.25(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_34.42.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<64x1024x5xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_41.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_40.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.28(%arg0: tensor<64x5x1024x256xf32>, %arg1: tensor<f32>) -> tensor<64x5x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    return %0 : tensor<64x5x1024xf32>
  }
  func.func private @region_39.50.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.29(%arg0: tensor<64x5x1024x8xf32>, %arg1: tensor<f32>) -> tensor<64x5x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<64x5x1024x8xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    return %0 : tensor<64x5x1024xf32>
  }
  func.func private @region_38.48.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<64x256x5xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x5xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_36.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.31(%arg0: tensor<64x256x160xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_34.42.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<64x1024x5xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_35.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_34.42.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.34(%arg0: tensor<64x1024x5xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_32.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %0 : tensor<64x1024xf32>
  }
  func.func private @region_18.23.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.36(%arg0: tensor<64x4096x2xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_31.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.37(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_18.23.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.38(%arg0: tensor<64x4096x2xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_30.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.39(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_29.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.40(%arg0: tensor<64x2x4096x256xf32>, %arg1: tensor<f32>) -> tensor<64x2x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    return %0 : tensor<64x2x4096xf32>
  }
  func.func private @region_28.36.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.41(%arg0: tensor<64x2x4096x8xf32>, %arg1: tensor<f32>) -> tensor<64x2x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x4096x8xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    return %0 : tensor<64x2x4096xf32>
  }
  func.func private @region_22.29.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.42(%arg0: tensor<64x256x2xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x2xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_27.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.43(%arg0: tensor<64x256x64xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_18.23.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.44(%arg0: tensor<64x4096x2xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_26.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_18.23.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.46(%arg0: tensor<64x4096x2xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_25.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.47(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_24.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.48(%arg0: tensor<64x2x4096x256xf32>, %arg1: tensor<f32>) -> tensor<64x2x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    return %0 : tensor<64x2x4096xf32>
  }
  func.func private @region_23.31.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.49(%arg0: tensor<64x2x4096x8xf32>, %arg1: tensor<f32>) -> tensor<64x2x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x4096x8xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    return %0 : tensor<64x2x4096xf32>
  }
  func.func private @region_22.29.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.50(%arg0: tensor<64x256x2xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x2xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_20.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.51(%arg0: tensor<64x256x64xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_18.23.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.52(%arg0: tensor<64x4096x2xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_19.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.53(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_18.23.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.54(%arg0: tensor<64x4096x2xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_16.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.55(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    return %0 : tensor<64x4096xf32>
  }
  func.func private @region_15.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.56(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<f32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    return %0 : tensor<64x16384xf32>
  }
  func.func private @region_14.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.57(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<f32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    return %0 : tensor<64x16384xf32>
  }
  func.func private @region_13.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.58(%arg0: tensor<64x1x16384x256xf32>, %arg1: tensor<f32>) -> tensor<64x1x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
    return %0 : tensor<64x1x16384xf32>
  }
  func.func private @region_12.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_12.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.59(%arg0: tensor<f32>, %arg1: tensor<64x1x16384x256xf32>) -> tensor<64x1x16384xf32> {
    %0 = "stablehlo.reduce_window"(%arg1, %arg0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384x8xf32>
    %1 = stablehlo.reduce(%0 init: %arg0) applies stablehlo.maximum across dimensions = [3] : (tensor<64x1x16384x8xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
    return %1 : tensor<64x1x16384xf32>
  }
  func.func private @region_11.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.60(%arg0: tensor<64x256x32xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_10.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.61(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<f32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    return %0 : tensor<64x16384xf32>
  }
  func.func private @region_9.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.62(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<f32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    return %0 : tensor<64x16384xf32>
  }
  func.func private @region_8.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.63(%arg0: tensor<64x1x16384x256xf32>, %arg1: tensor<f32>) -> tensor<64x1x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
    return %0 : tensor<64x1x16384xf32>
  }
  func.func private @region_7.12.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_7.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.64(%arg0: tensor<f32>, %arg1: tensor<64x1x16384x256xf32>) -> tensor<64x1x16384xf32> {
    %0 = "stablehlo.reduce_window"(%arg1, %arg0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384x8xf32>
    %1 = stablehlo.reduce(%0 init: %arg0) applies stablehlo.maximum across dimensions = [3] : (tensor<64x1x16384x8xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
    return %1 : tensor<64x1x16384xf32>
  }
  func.func private @region_4.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.65(%arg0: tensor<64x256x32xf32>, %arg1: tensor<f32>) -> tensor<64x256xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %0 : tensor<64x256xf32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.66(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<f32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    return %0 : tensor<64x16384xf32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.67(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<f32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    return %0 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<64x128x128x150xf32>, %arg1: tensor<150xf32>) -> tensor<64x150x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<150xf32>) -> tensor<64x128x128x150xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x128x128x150xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,150,128,128]{1,3,2,0}"} : (tensor<64x128x128x150xf32>) -> tensor<64x150x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x150x128x128xf32>
    return %3 : tensor<64x150x128x128xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,150]{1,0,2,3}"} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x150xf32>
    return %1 : tensor<1x1x256x150xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x128x128x256xf32>, %arg4: tensor<256xf32>) -> tensor<64x128x128x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x128x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x128x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x128x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x128x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x256xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<64x128x128x256xf32>
    return %9 : tensor<64x128x128x256xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x1024x256xf32>
    return %1 : tensor<1x1x1024x256xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<128x128x2xi32>, %arg1: tensor<128x128x2xi32>, %arg2: tensor<64x16384x256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<128x128x2xi32>, %arg5: tensor<128x128x2xi32>, %arg6: tensor<128xf32>, %arg7: tensor<64x4096x256xf32>, %arg8: tensor<256xf32>, %arg9: tensor<128x128x2xi32>, %arg10: tensor<128x128x2xi32>, %arg11: tensor<64x1024x256xf32>, %arg12: tensor<256xf32>, %arg13: tensor<f32>, %arg14: tensor<128x128x2xi32>, %arg15: tensor<128x128x2xi32>, %arg16: tensor<i32>, %arg17: tensor<f32>, %arg18: tensor<64x256x256xf32>, %arg19: tensor<256xf32>) -> tensor<64x128x128x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %1 = stablehlo.add %arg18, %0 : tensor<64x256x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %3 = mhlo.copy %2 : tensor<64x256x256xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<64x256x16x16xf32>
    %5 = mhlo.bitcast %arg15 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %6 = "stablehlo.gather"(%4, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %9 = mhlo.copy %8 : tensor<64x256x128x128xf32>
    %10 = stablehlo.broadcast_in_dim %arg13, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %12 = stablehlo.multiply %arg6, %11 : tensor<128xf32>
    %13 = stablehlo.broadcast_in_dim %arg17, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %14 = stablehlo.add %12, %13 : tensor<128xf32>
    %15 = stablehlo.maximum %10, %14 : tensor<128xf32>
    %16 = mhlo.bitcast %15 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %17 = stablehlo.convert %16 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %18 = stablehlo.broadcast_in_dim %arg16, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c = stablehlo.constant dense<16> : tensor<i32>
    %20 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %21 = stablehlo.add %17, %20 : tensor<128x1xi32>
    %22 = stablehlo.select %19, %21, %17 : tensor<128x1xi1>, tensor<128x1xi32>
    %23 = mhlo.bitcast %22 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %c_0 = stablehlo.constant dense<15> : tensor<i32>
    %25 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %26 = stablehlo.convert %15 : (tensor<128xf32>) -> tensor<128xi32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %28 = stablehlo.add %26, %27 : tensor<128xi32>
    %29 = stablehlo.minimum %25, %28 : tensor<128xi32>
    %30 = stablehlo.broadcast_in_dim %arg16, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %31 = stablehlo.compare LT, %29, %30 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %32 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %33 = stablehlo.add %29, %32 : tensor<128xi32>
    %34 = stablehlo.select %31, %33, %29 : tensor<128xi1>, tensor<128xi32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %36 = stablehlo.concatenate %24, %35, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %38 = "stablehlo.gather"(%4, %37) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %40 = stablehlo.transpose %39, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %41 = mhlo.copy %40 : tensor<64x256x128x128xf32>
    %42 = stablehlo.subtract %41, %9 : tensor<64x256x128x128xf32>
    %43 = stablehlo.convert %26 : (tensor<128xi32>) -> tensor<128xf32>
    %44 = stablehlo.subtract %15, %43 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %45 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %46 = stablehlo.clamp %10, %44, %45 : tensor<128xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %48 = stablehlo.multiply %42, %47 : tensor<64x256x128x128xf32>
    %49 = stablehlo.add %9, %48 : tensor<64x256x128x128xf32>
    %50 = mhlo.bitcast %arg14 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %51 = "stablehlo.gather"(%4, %50) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %53 = stablehlo.transpose %52, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %54 = mhlo.copy %53 : tensor<64x256x128x128xf32>
    %55 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %56 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %57 = stablehlo.add %17, %56 : tensor<128x1xi32>
    %58 = stablehlo.minimum %55, %57 : tensor<128x1xi32>
    %59 = stablehlo.compare LT, %58, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %60 = stablehlo.add %58, %20 : tensor<128x1xi32>
    %61 = stablehlo.select %59, %60, %58 : tensor<128x1xi1>, tensor<128x1xi32>
    %62 = mhlo.bitcast %61 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %64 = stablehlo.concatenate %63, %35, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %66 = "stablehlo.gather"(%4, %65) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %68 = stablehlo.transpose %67, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %69 = mhlo.copy %68 : tensor<64x256x128x128xf32>
    %70 = stablehlo.subtract %69, %54 : tensor<64x256x128x128xf32>
    %71 = stablehlo.multiply %70, %47 : tensor<64x256x128x128xf32>
    %72 = stablehlo.add %54, %71 : tensor<64x256x128x128xf32>
    %73 = stablehlo.subtract %72, %49 : tensor<64x256x128x128xf32>
    %74 = stablehlo.broadcast_in_dim %arg13, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %75 = stablehlo.convert %17 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %76 = stablehlo.subtract %16, %75 : tensor<128x1xf32>
    %77 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %78 = stablehlo.clamp %74, %76, %77 : tensor<128x1xf32>
    %79 = mhlo.bitcast %78 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %81 = stablehlo.multiply %73, %80 : tensor<64x256x128x128xf32>
    %82 = stablehlo.add %49, %81 : tensor<64x256x128x128xf32>
    %83 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<256xf32>) -> tensor<64x1024x256xf32>
    %84 = stablehlo.add %arg11, %83 : tensor<64x1024x256xf32>
    %85 = stablehlo.transpose %84, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %86 = mhlo.copy %85 : tensor<64x256x1024xf32>
    %87 = mhlo.bitcast %86 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<64x256x32x32xf32>
    %88 = mhlo.bitcast %arg10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %89 = "stablehlo.gather"(%87, %88) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %90 = mhlo.bitcast %89 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %91 = stablehlo.transpose %90, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %92 = mhlo.copy %91 : tensor<64x256x128x128xf32>
    %cst_3 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %94 = stablehlo.multiply %arg6, %93 : tensor<128xf32>
    %95 = stablehlo.add %94, %13 : tensor<128xf32>
    %96 = stablehlo.maximum %10, %95 : tensor<128xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %98 = stablehlo.convert %97 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %99 = stablehlo.compare LT, %98, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_4 = stablehlo.constant dense<32> : tensor<i32>
    %100 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %101 = stablehlo.add %98, %100 : tensor<128x1xi32>
    %102 = stablehlo.select %99, %101, %98 : tensor<128x1xi1>, tensor<128x1xi32>
    %103 = mhlo.bitcast %102 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %104 = stablehlo.broadcast_in_dim %103, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %c_5 = stablehlo.constant dense<31> : tensor<i32>
    %105 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %106 = stablehlo.convert %96 : (tensor<128xf32>) -> tensor<128xi32>
    %107 = stablehlo.add %106, %27 : tensor<128xi32>
    %108 = stablehlo.minimum %105, %107 : tensor<128xi32>
    %109 = stablehlo.compare LT, %108, %30 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %110 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %111 = stablehlo.add %108, %110 : tensor<128xi32>
    %112 = stablehlo.select %109, %111, %108 : tensor<128xi1>, tensor<128xi32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %114 = stablehlo.concatenate %104, %113, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %116 = "stablehlo.gather"(%87, %115) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %118 = stablehlo.transpose %117, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %119 = mhlo.copy %118 : tensor<64x256x128x128xf32>
    %120 = stablehlo.subtract %119, %92 : tensor<64x256x128x128xf32>
    %121 = stablehlo.convert %106 : (tensor<128xi32>) -> tensor<128xf32>
    %122 = stablehlo.subtract %96, %121 : tensor<128xf32>
    %123 = stablehlo.clamp %10, %122, %45 : tensor<128xf32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %125 = stablehlo.multiply %120, %124 : tensor<64x256x128x128xf32>
    %126 = stablehlo.add %92, %125 : tensor<64x256x128x128xf32>
    %127 = mhlo.bitcast %arg9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %128 = "stablehlo.gather"(%87, %127) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %129 = mhlo.bitcast %128 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %130 = stablehlo.transpose %129, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %131 = mhlo.copy %130 : tensor<64x256x128x128xf32>
    %132 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %133 = stablehlo.add %98, %56 : tensor<128x1xi32>
    %134 = stablehlo.minimum %132, %133 : tensor<128x1xi32>
    %135 = stablehlo.compare LT, %134, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %136 = stablehlo.add %134, %100 : tensor<128x1xi32>
    %137 = stablehlo.select %135, %136, %134 : tensor<128x1xi1>, tensor<128x1xi32>
    %138 = mhlo.bitcast %137 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %140 = stablehlo.concatenate %139, %113, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %141 = mhlo.bitcast %140 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %142 = "stablehlo.gather"(%87, %141) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %143 = mhlo.bitcast %142 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %144 = stablehlo.transpose %143, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %145 = mhlo.copy %144 : tensor<64x256x128x128xf32>
    %146 = stablehlo.subtract %145, %131 : tensor<64x256x128x128xf32>
    %147 = stablehlo.multiply %146, %124 : tensor<64x256x128x128xf32>
    %148 = stablehlo.add %131, %147 : tensor<64x256x128x128xf32>
    %149 = stablehlo.subtract %148, %126 : tensor<64x256x128x128xf32>
    %150 = stablehlo.convert %98 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %151 = stablehlo.subtract %97, %150 : tensor<128x1xf32>
    %152 = stablehlo.clamp %74, %151, %77 : tensor<128x1xf32>
    %153 = mhlo.bitcast %152 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %155 = stablehlo.multiply %149, %154 : tensor<64x256x128x128xf32>
    %156 = stablehlo.add %126, %155 : tensor<64x256x128x128xf32>
    %157 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<256xf32>) -> tensor<64x4096x256xf32>
    %158 = stablehlo.add %arg7, %157 : tensor<64x4096x256xf32>
    %159 = stablehlo.transpose %158, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,4096]{1,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
    %160 = mhlo.copy %159 : tensor<64x256x4096xf32>
    %161 = mhlo.bitcast %160 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
    %162 = mhlo.bitcast %arg5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %163 = "stablehlo.gather"(%161, %162) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %165 = stablehlo.transpose %164, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %166 = mhlo.copy %165 : tensor<64x256x128x128xf32>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %167 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %168 = stablehlo.multiply %arg6, %167 : tensor<128xf32>
    %169 = stablehlo.add %168, %13 : tensor<128xf32>
    %170 = stablehlo.maximum %10, %169 : tensor<128xf32>
    %171 = mhlo.bitcast %170 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %172 = stablehlo.convert %171 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %173 = stablehlo.compare LT, %172, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_7 = stablehlo.constant dense<64> : tensor<i32>
    %174 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %175 = stablehlo.add %172, %174 : tensor<128x1xi32>
    %176 = stablehlo.select %173, %175, %172 : tensor<128x1xi1>, tensor<128x1xi32>
    %177 = mhlo.bitcast %176 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %c_8 = stablehlo.constant dense<63> : tensor<i32>
    %179 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %180 = stablehlo.convert %170 : (tensor<128xf32>) -> tensor<128xi32>
    %181 = stablehlo.add %180, %27 : tensor<128xi32>
    %182 = stablehlo.minimum %179, %181 : tensor<128xi32>
    %183 = stablehlo.compare LT, %182, %30 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %184 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %185 = stablehlo.add %182, %184 : tensor<128xi32>
    %186 = stablehlo.select %183, %185, %182 : tensor<128xi1>, tensor<128xi32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %188 = stablehlo.concatenate %178, %187, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %189 = mhlo.bitcast %188 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %190 = "stablehlo.gather"(%161, %189) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %191 = mhlo.bitcast %190 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %192 = stablehlo.transpose %191, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %193 = mhlo.copy %192 : tensor<64x256x128x128xf32>
    %194 = stablehlo.subtract %193, %166 : tensor<64x256x128x128xf32>
    %195 = stablehlo.convert %180 : (tensor<128xi32>) -> tensor<128xf32>
    %196 = stablehlo.subtract %170, %195 : tensor<128xf32>
    %197 = stablehlo.clamp %10, %196, %45 : tensor<128xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %199 = stablehlo.multiply %194, %198 : tensor<64x256x128x128xf32>
    %200 = stablehlo.add %166, %199 : tensor<64x256x128x128xf32>
    %201 = mhlo.bitcast %arg4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %202 = "stablehlo.gather"(%161, %201) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %203 = mhlo.bitcast %202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %204 = stablehlo.transpose %203, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %205 = mhlo.copy %204 : tensor<64x256x128x128xf32>
    %206 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %207 = stablehlo.add %172, %56 : tensor<128x1xi32>
    %208 = stablehlo.minimum %206, %207 : tensor<128x1xi32>
    %209 = stablehlo.compare LT, %208, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %210 = stablehlo.add %208, %174 : tensor<128x1xi32>
    %211 = stablehlo.select %209, %210, %208 : tensor<128x1xi1>, tensor<128x1xi32>
    %212 = mhlo.bitcast %211 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %214 = stablehlo.concatenate %213, %187, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %215 = mhlo.bitcast %214 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %216 = "stablehlo.gather"(%161, %215) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %217 = mhlo.bitcast %216 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %218 = stablehlo.transpose %217, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %219 = mhlo.copy %218 : tensor<64x256x128x128xf32>
    %220 = stablehlo.subtract %219, %205 : tensor<64x256x128x128xf32>
    %221 = stablehlo.multiply %220, %198 : tensor<64x256x128x128xf32>
    %222 = stablehlo.add %205, %221 : tensor<64x256x128x128xf32>
    %223 = stablehlo.subtract %222, %200 : tensor<64x256x128x128xf32>
    %224 = stablehlo.convert %172 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %225 = stablehlo.subtract %171, %224 : tensor<128x1xf32>
    %226 = stablehlo.clamp %74, %225, %77 : tensor<128x1xf32>
    %227 = mhlo.bitcast %226 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %229 = stablehlo.multiply %223, %228 : tensor<64x256x128x128xf32>
    %230 = stablehlo.add %200, %229 : tensor<64x256x128x128xf32>
    %231 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<256xf32>) -> tensor<64x16384x256xf32>
    %232 = stablehlo.add %arg2, %231 : tensor<64x16384x256xf32>
    %233 = stablehlo.transpose %232, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,16384]{1,2,0}"} : (tensor<64x16384x256xf32>) -> tensor<64x256x16384xf32>
    %234 = mhlo.copy %233 : tensor<64x256x16384xf32>
    %235 = mhlo.bitcast %234 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x16384xf32>) -> tensor<64x256x128x128xf32>
    %236 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %237 = "stablehlo.gather"(%235, %236) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %239 = stablehlo.transpose %238, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %240 = mhlo.copy %239 : tensor<64x256x128x128xf32>
    %241 = stablehlo.iota dim = 0 : tensor<128xi32>
    %242 = stablehlo.convert %241 : (tensor<128xi32>) -> tensor<128xf32>
    %243 = stablehlo.maximum %10, %242 : tensor<128xf32>
    %244 = mhlo.bitcast %243 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %245 = stablehlo.convert %244 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %246 = stablehlo.compare LT, %245, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_9 = stablehlo.constant dense<128> : tensor<i32>
    %247 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %248 = stablehlo.add %245, %247 : tensor<128x1xi32>
    %249 = stablehlo.select %246, %248, %245 : tensor<128x1xi1>, tensor<128x1xi32>
    %250 = mhlo.bitcast %249 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %c_10 = stablehlo.constant dense<127> : tensor<i32>
    %252 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %253 = stablehlo.convert %243 : (tensor<128xf32>) -> tensor<128xi32>
    %254 = stablehlo.add %253, %27 : tensor<128xi32>
    %255 = stablehlo.minimum %252, %254 : tensor<128xi32>
    %256 = stablehlo.compare LT, %255, %30 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %257 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %258 = stablehlo.add %255, %257 : tensor<128xi32>
    %259 = stablehlo.select %256, %258, %255 : tensor<128xi1>, tensor<128xi32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %261 = stablehlo.concatenate %251, %260, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %262 = mhlo.bitcast %261 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %263 = "stablehlo.gather"(%235, %262) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %264 = mhlo.bitcast %263 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %265 = stablehlo.transpose %264, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %266 = mhlo.copy %265 : tensor<64x256x128x128xf32>
    %267 = stablehlo.subtract %266, %240 : tensor<64x256x128x128xf32>
    %268 = stablehlo.convert %253 : (tensor<128xi32>) -> tensor<128xf32>
    %269 = stablehlo.subtract %243, %268 : tensor<128xf32>
    %270 = stablehlo.clamp %10, %269, %45 : tensor<128xf32>
    %271 = stablehlo.broadcast_in_dim %270, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %272 = stablehlo.multiply %267, %271 : tensor<64x256x128x128xf32>
    %273 = stablehlo.add %240, %272 : tensor<64x256x128x128xf32>
    %274 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %275 = "stablehlo.gather"(%235, %274) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %276 = mhlo.bitcast %275 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %277 = stablehlo.transpose %276, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %278 = mhlo.copy %277 : tensor<64x256x128x128xf32>
    %279 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %280 = stablehlo.add %245, %56 : tensor<128x1xi32>
    %281 = stablehlo.minimum %279, %280 : tensor<128x1xi32>
    %282 = stablehlo.compare LT, %281, %18 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %283 = stablehlo.add %281, %247 : tensor<128x1xi32>
    %284 = stablehlo.select %282, %283, %281 : tensor<128x1xi1>, tensor<128x1xi32>
    %285 = mhlo.bitcast %284 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %287 = stablehlo.concatenate %286, %260, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
    %289 = "stablehlo.gather"(%235, %288) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
    %290 = mhlo.bitcast %289 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
    %291 = stablehlo.transpose %290, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
    %292 = mhlo.copy %291 : tensor<64x256x128x128xf32>
    %293 = stablehlo.subtract %292, %278 : tensor<64x256x128x128xf32>
    %294 = stablehlo.multiply %293, %271 : tensor<64x256x128x128xf32>
    %295 = stablehlo.add %278, %294 : tensor<64x256x128x128xf32>
    %296 = stablehlo.subtract %295, %273 : tensor<64x256x128x128xf32>
    %297 = stablehlo.convert %245 : (tensor<128x1xi32>) -> tensor<128x1xf32>
    %298 = stablehlo.subtract %244, %297 : tensor<128x1xf32>
    %299 = stablehlo.clamp %74, %298, %77 : tensor<128x1xf32>
    %300 = mhlo.bitcast %299 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
    %302 = stablehlo.multiply %296, %301 : tensor<64x256x128x128xf32>
    %303 = stablehlo.add %273, %302 : tensor<64x256x128x128xf32>
    %304 = stablehlo.concatenate %82, %156, %230, %303, dim = 1 : (tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>) -> tensor<64x1024x128x128xf32>
    %305 = stablehlo.transpose %304, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,1024]{2,1,3,0}"} : (tensor<64x1024x128x128xf32>) -> tensor<64x128x128x1024xf32>
    %306 = mhlo.copy %305 : tensor<64x128x128x1024xf32>
    return %306 : tensor<64x128x128x1024xf32>
  }
  func.func private @fused_computation.74() -> tensor<128x128x2xi32> {
    %c = stablehlo.constant dense<127> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %4 = stablehlo.maximum %1, %3 : tensor<128xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %6 = stablehlo.convert %5 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %8 = stablehlo.add %6, %7 : tensor<128x1xi32>
    %9 = stablehlo.minimum %0, %8 : tensor<128x1xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_2 = stablehlo.constant dense<128> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %13 = stablehlo.add %9, %12 : tensor<128x1xi32>
    %14 = stablehlo.select %11, %13, %9 : tensor<128x1xi1>, tensor<128x1xi32>
    %15 = mhlo.bitcast %14 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %17 = stablehlo.convert %4 : (tensor<128xf32>) -> tensor<128xi32>
    %18 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %20 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %21 = stablehlo.add %17, %20 : tensor<128xi32>
    %22 = stablehlo.select %19, %21, %17 : tensor<128xi1>, tensor<128xi32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %24 = stablehlo.concatenate %16, %23, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %24 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.75() -> tensor<128x128x2xi32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %3 = stablehlo.maximum %0, %2 : tensor<128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %5 = stablehlo.convert %4 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %7 = stablehlo.compare LT, %5, %6 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_0 = stablehlo.constant dense<128> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %9 = stablehlo.add %5, %8 : tensor<128x1xi32>
    %10 = stablehlo.select %7, %9, %5 : tensor<128x1xi1>, tensor<128x1xi32>
    %11 = mhlo.bitcast %10 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %13 = stablehlo.convert %3 : (tensor<128xf32>) -> tensor<128xi32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %15 = stablehlo.compare LT, %13, %14 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %16 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %17 = stablehlo.add %13, %16 : tensor<128xi32>
    %18 = stablehlo.select %15, %17, %13 : tensor<128xi1>, tensor<128xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %20 = stablehlo.concatenate %12, %19, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %20 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.76(%arg0: tensor<256x32xf32>) -> tensor<64x32x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<256x32xf32>) -> tensor<32x256xf32>
    %1 = mhlo.copy %0 : tensor<32x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<64x32x256xf32>
    return %2 : tensor<64x32x256xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384x32xf32>, %arg4: tensor<64x16384xf32>) -> tensor<64x32x16384xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384x32xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<64x128x128x32xf32>
    %11 = stablehlo.transpose %10, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %12 = mhlo.copy %11 : tensor<64x32x128x128xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    return %13 : tensor<64x32x16384xf32>
  }
  func.func private @fused_computation.78() -> tensor<128x128x2xi32> {
    %c = stablehlo.constant dense<63> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %6 = stablehlo.multiply %5, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.add %6, %7 : tensor<128xf32>
    %9 = stablehlo.maximum %1, %8 : tensor<128xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %11 = stablehlo.convert %10 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %13 = stablehlo.add %11, %12 : tensor<128x1xi32>
    %14 = stablehlo.minimum %0, %13 : tensor<128x1xi32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %15 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %16 = stablehlo.compare LT, %14, %15 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_4 = stablehlo.constant dense<64> : tensor<i32>
    %17 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %18 = stablehlo.add %14, %17 : tensor<128x1xi32>
    %19 = stablehlo.select %16, %18, %14 : tensor<128x1xi1>, tensor<128x1xi32>
    %20 = mhlo.bitcast %19 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %22 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %23 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %24 = stablehlo.compare LT, %22, %23 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %25 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %26 = stablehlo.add %22, %25 : tensor<128xi32>
    %27 = stablehlo.select %24, %26, %22 : tensor<128xi1>, tensor<128xi32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %29 = stablehlo.concatenate %21, %28, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %29 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.79() -> tensor<128x128x2xi32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %2, %3 : tensor<128xf32>
    %5 = stablehlo.multiply %4, %3 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.add %5, %6 : tensor<128xf32>
    %8 = stablehlo.maximum %0, %7 : tensor<128xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %10 = stablehlo.convert %9 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %12 = stablehlo.compare LT, %10, %11 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_2 = stablehlo.constant dense<64> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %14 = stablehlo.add %10, %13 : tensor<128x1xi32>
    %15 = stablehlo.select %12, %14, %10 : tensor<128x1xi1>, tensor<128x1xi32>
    %16 = mhlo.bitcast %15 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %18 = stablehlo.convert %8 : (tensor<128xf32>) -> tensor<128xi32>
    %19 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %20 = stablehlo.compare LT, %18, %19 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %21 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %22 = stablehlo.add %18, %21 : tensor<128xi32>
    %23 = stablehlo.select %20, %22, %18 : tensor<128xi1>, tensor<128xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %25 = stablehlo.concatenate %17, %24, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %25 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.80(%arg0: tensor<256x64xf32>) -> tensor<64x64x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[64,256]{0,1}"} : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %1 = mhlo.copy %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
    return %2 : tensor<64x64x256xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096x64xf32>, %arg4: tensor<64x4096xf32>) -> tensor<64x64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<64x64x64x64xf32>
    %11 = stablehlo.transpose %10, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %12 = mhlo.copy %11 : tensor<64x64x64x64xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    return %13 : tensor<64x64x4096xf32>
  }
  func.func private @fused_computation.82() -> tensor<128x128x2xi32> {
    %c = stablehlo.constant dense<31> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %9 = stablehlo.add %7, %8 : tensor<128xf32>
    %10 = stablehlo.maximum %1, %9 : tensor<128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %12 = stablehlo.convert %11 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %14 = stablehlo.add %12, %13 : tensor<128x1xi32>
    %15 = stablehlo.minimum %0, %14 : tensor<128x1xi32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %17 = stablehlo.compare LT, %15, %16 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_5 = stablehlo.constant dense<32> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %19 = stablehlo.add %15, %18 : tensor<128x1xi32>
    %20 = stablehlo.select %17, %19, %15 : tensor<128x1xi1>, tensor<128x1xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %23 = stablehlo.convert %10 : (tensor<128xf32>) -> tensor<128xi32>
    %24 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %25 = stablehlo.compare LT, %23, %24 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %26 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %27 = stablehlo.add %23, %26 : tensor<128xi32>
    %28 = stablehlo.select %25, %27, %23 : tensor<128xi1>, tensor<128xi32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %30 = stablehlo.concatenate %22, %29, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %30 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.83() -> tensor<128x128x2xi32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %2, %3 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.add %6, %7 : tensor<128xf32>
    %9 = stablehlo.maximum %0, %8 : tensor<128xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %11 = stablehlo.convert %10 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %13 = stablehlo.compare LT, %11, %12 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_3 = stablehlo.constant dense<32> : tensor<i32>
    %14 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %15 = stablehlo.add %11, %14 : tensor<128x1xi32>
    %16 = stablehlo.select %13, %15, %11 : tensor<128x1xi1>, tensor<128x1xi32>
    %17 = mhlo.bitcast %16 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %19 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %20 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %21 = stablehlo.compare LT, %19, %20 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %22 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %23 = stablehlo.add %19, %22 : tensor<128xi32>
    %24 = stablehlo.select %21, %23, %19 : tensor<128xi1>, tensor<128xi32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %26 = stablehlo.concatenate %18, %25, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %26 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.84(%arg0: tensor<256x160xf32>) -> tensor<64x160x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[160,256]{0,1}"} : (tensor<256x160xf32>) -> tensor<160x256xf32>
    %1 = mhlo.copy %0 : tensor<160x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<64x160x256xf32>
    return %2 : tensor<64x160x256xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024x160xf32>, %arg4: tensor<64x1024xf32>) -> tensor<64x160x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024x160xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<64x32x32x160xf32>
    %11 = stablehlo.transpose %10, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %12 = mhlo.copy %11 : tensor<64x160x32x32xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    return %13 : tensor<64x160x1024xf32>
  }
  func.func private @fused_computation.86() -> tensor<128x128x2xi32> {
    %c = stablehlo.constant dense<15> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %2 = stablehlo.iota dim = 0 : tensor<128xi32>
    %3 = stablehlo.convert %2 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %5 = stablehlo.add %3, %4 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %9 = stablehlo.add %7, %8 : tensor<128xf32>
    %10 = stablehlo.maximum %1, %9 : tensor<128xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %12 = stablehlo.convert %11 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %14 = stablehlo.add %12, %13 : tensor<128x1xi32>
    %15 = stablehlo.minimum %0, %14 : tensor<128x1xi32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %17 = stablehlo.compare LT, %15, %16 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_5 = stablehlo.constant dense<16> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %19 = stablehlo.add %15, %18 : tensor<128x1xi32>
    %20 = stablehlo.select %17, %19, %15 : tensor<128x1xi1>, tensor<128x1xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %23 = stablehlo.convert %10 : (tensor<128xf32>) -> tensor<128xi32>
    %24 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %25 = stablehlo.compare LT, %23, %24 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %26 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %27 = stablehlo.add %23, %26 : tensor<128xi32>
    %28 = stablehlo.select %25, %27, %23 : tensor<128xi1>, tensor<128xi32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %30 = stablehlo.concatenate %22, %29, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %30 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.87() -> tensor<128x128x2xi32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.iota dim = 0 : tensor<128xi32>
    %2 = stablehlo.convert %1 : (tensor<128xi32>) -> tensor<128xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %4 = stablehlo.add %2, %3 : tensor<128xf32>
    %cst_1 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<128xf32>
    %cst_2 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %8 = stablehlo.add %6, %7 : tensor<128xf32>
    %9 = stablehlo.maximum %0, %8 : tensor<128xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
    %11 = stablehlo.convert %10 : (tensor<128x1xf32>) -> tensor<128x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %13 = stablehlo.compare LT, %11, %12 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
    %c_3 = stablehlo.constant dense<16> : tensor<i32>
    %14 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
    %15 = stablehlo.add %11, %14 : tensor<128x1xi32>
    %16 = stablehlo.select %13, %15, %11 : tensor<128x1xi1>, tensor<128x1xi32>
    %17 = mhlo.bitcast %16 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %19 = stablehlo.convert %9 : (tensor<128xf32>) -> tensor<128xi32>
    %20 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %21 = stablehlo.compare LT, %19, %20 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
    %22 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %23 = stablehlo.add %19, %22 : tensor<128xi32>
    %24 = stablehlo.select %21, %23, %19 : tensor<128xi1>, tensor<128xi32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
    %26 = stablehlo.concatenate %18, %25, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
    return %26 : tensor<128x128x2xi32>
  }
  func.func private @fused_computation.88() -> tensor<128xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<128xi32>
    %1 = stablehlo.convert %0 : (tensor<128xi32>) -> tensor<128xf32>
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.add %1, %2 : tensor<128xf32>
    return %3 : tensor<128xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<256x256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = mhlo.copy %0 : tensor<256x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<64x256x256xf32>
    return %2 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256x256xf32>, %arg4: tensor<64x256xf32>) -> tensor<64x256x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x256x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<64x16x16x256xf32>
    %11 = stablehlo.transpose %10, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %12 = mhlo.copy %11 : tensor<64x256x16x16xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    return %13 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<64x256x256xf32>, %arg1: tensor<64x256xf32>) -> tensor<64x256x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x256x256xf32>
    return %4 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<64x256x256xf32>, %arg1: tensor<16384x256xf32>, %arg2: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x256x256xf32>
    return %3 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<64x16x16x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<16384x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<64x16x16x1024xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x16x16x1024xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x16x16x1024xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
    %5 = mhlo.copy %4 : tensor<64x1024x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %8 = mhlo.copy %7 : tensor<64x256x1024xf32>
    %9 = stablehlo.negate %2 : tensor<64x16x16x1024xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x16x16x1024xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
    %13 = mhlo.copy %12 : tensor<64x1024x16x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %16 = mhlo.copy %15 : tensor<64x256x1024xf32>
    %17 = stablehlo.abs %16 : tensor<64x256x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x256x1024xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x256x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x256x1024xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x256x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x256x1024xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x256x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x256x1024xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x256x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x256x1024xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x256x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x256x1024xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x256x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x256x1024xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x256x1024xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x256x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %45 = stablehlo.negate %21 : tensor<64x256x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %48 = stablehlo.exponential %45 : tensor<64x256x1024xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x256x1024xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x256x1024xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x256x1024xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x256x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x256x1024xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x256x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x256x1024xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x256x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x256x1024xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x256x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x256x1024xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x256x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x256x1024xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x256x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x256x1024xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x256x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x256x1024xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x256x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x256x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x256x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x256x1024xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x256x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x256x1024xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x256x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x256x1024xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x256x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x256x1024xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x256x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x256x1024xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x256x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x256x1024xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x256x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x256x1024xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x256x1024xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x256x1024xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x256x1024xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
    return %107 : tensor<16384x1024xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x1024xf32>
    return %1 : tensor<3x3x1x1024xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<16384x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<64x16x16x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,256]{1,2,0}"} : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
    %4 = mhlo.copy %3 : tensor<64x1024x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x256xf32>) -> tensor<64x1024x16x16xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,16,16,1024]{2,1,3,0}"} : (tensor<64x1024x16x16xf32>) -> tensor<64x16x16x1024xf32>
    %7 = mhlo.copy %6 : tensor<64x16x16x1024xf32>
    return %7 : tensor<64x16x16x1024xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256x256xf32>, %arg4: tensor<64x256xf32>) -> tensor<16384x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x256x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    return %10 : tensor<16384x256xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<64x256x256xf32>, %arg1: tensor<64x256xf32>) -> tensor<64x256x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x256x256xf32>
    return %4 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<64x256x256xf32>, %arg1: tensor<16384x256xf32>, %arg2: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x256x256xf32>
    return %3 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<512x256x32xf32>) -> tensor<16384x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,8,32]{3,1,2,0}"} : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
    %2 = mhlo.copy %1 : tensor<64x256x8x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x8x32xf32>) -> tensor<16384x256xf32>
    return %3 : tensor<16384x256xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x8x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    return %5 : tensor<512x256x32xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<64x8x256x256xf32>, %arg1: tensor<64x8x256xf32>) -> tensor<512x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<64x8x256x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x256xf32>) -> tensor<512x256x256xf32>
    return %2 : tensor<512x256x256xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<64x8x256xf32>, %arg1: tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<512x256x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x8x256xf32>
    %4 = stablehlo.maximum %3, %arg0 : tensor<64x8x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x8x256x256xf32>
    %7 = stablehlo.exponential %6 : tensor<64x8x256x256xf32>
    return %7 : tensor<64x8x256x256xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x256x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    return %2 : tensor<64x8x256x256xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,32,256]{2,1,3,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
    %4 = mhlo.copy %3 : tensor<64x8x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x32x256xf32>) -> tensor<512x32x256xf32>
    return %5 : tensor<512x32x256xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x8x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    return %5 : tensor<512x256x32xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256x256xf32>, %arg4: tensor<64x256xf32>) -> tensor<16384x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x256x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    return %10 : tensor<16384x256xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<64x256x256xf32>, %arg1: tensor<64x256xf32>) -> tensor<64x256x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x256x256xf32>
    return %4 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<64x256x256xf32>, %arg1: tensor<16384x256xf32>, %arg2: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x256x256xf32>
    return %3 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<64x16x16x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<16384x1024xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<1024xf32>) -> tensor<64x16x16x1024xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x16x16x1024xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x16x16x1024xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
    %5 = mhlo.copy %4 : tensor<64x1024x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %8 = mhlo.copy %7 : tensor<64x256x1024xf32>
    %9 = stablehlo.negate %2 : tensor<64x16x16x1024xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x16x16x1024xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
    %13 = mhlo.copy %12 : tensor<64x1024x16x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
    %16 = mhlo.copy %15 : tensor<64x256x1024xf32>
    %17 = stablehlo.abs %16 : tensor<64x256x1024xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x256x1024xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x256x1024xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x256x1024xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x256x1024xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x256x1024xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x256x1024xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x256x1024xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x256x1024xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x256x1024xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x256x1024xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x256x1024xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x256x1024xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x256x1024xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x256x1024xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x256x1024xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %45 = stablehlo.negate %21 : tensor<64x256x1024xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %48 = stablehlo.exponential %45 : tensor<64x256x1024xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x256x1024xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x256x1024xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x256x1024xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x256x1024xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x256x1024xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x256x1024xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x256x1024xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x256x1024xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x256x1024xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x256x1024xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x256x1024xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x256x1024xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x256x1024xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x256x1024xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x256x1024xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x256x1024xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x256x1024xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x256x1024xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x256x1024xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x256x1024xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x256x1024xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x256x1024xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x256x1024xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x256x1024xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x256x1024xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x256x1024xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x256x1024xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x256x1024xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x256x1024xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x256x1024xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x256x1024xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x256x1024xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x256x1024xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x256x1024xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x256x1024xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x256x1024xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
    return %107 : tensor<16384x1024xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x1024xf32>
    return %1 : tensor<3x3x1x1024xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<16384x1024xf32>, %arg1: tensor<1024xf32>) -> tensor<64x16x16x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x1024xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,256]{1,2,0}"} : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
    %4 = mhlo.copy %3 : tensor<64x1024x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x256xf32>) -> tensor<64x1024x16x16xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,16,16,1024]{2,1,3,0}"} : (tensor<64x1024x16x16xf32>) -> tensor<64x16x16x1024xf32>
    %7 = mhlo.copy %6 : tensor<64x16x16x1024xf32>
    return %7 : tensor<64x16x16x1024xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256x256xf32>, %arg4: tensor<64x256xf32>) -> tensor<16384x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x256x256xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x256x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    return %10 : tensor<16384x256xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<64x256x256xf32>, %arg1: tensor<64x256xf32>) -> tensor<64x256x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x256x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x256x256xf32>
    return %4 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x256xf32>, %arg5: tensor<64x16x16x256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<64x16x16x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %6 = mhlo.copy %5 : tensor<64x256x16x16xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %9 = mhlo.copy %8 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %11 = stablehlo.multiply %arg4, %10 : tensor<64x256xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %13 = stablehlo.subtract %9, %12 : tensor<64x256x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<64x256x256xf32>
    %16 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x256x256xf32>
    %18 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x256x256xf32>
    %20 = stablehlo.add %2, %19 : tensor<64x256x256xf32>
    return %20 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<512x256x32xf32>) -> tensor<16384x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,8,32]{3,1,2,0}"} : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
    %2 = mhlo.copy %1 : tensor<64x256x8x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x8x32xf32>) -> tensor<16384x256xf32>
    return %3 : tensor<16384x256xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x8x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    return %5 : tensor<512x256x32xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<64x8x256x256xf32>, %arg1: tensor<64x8x256xf32>) -> tensor<512x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<64x8x256x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x256xf32>) -> tensor<512x256x256xf32>
    return %2 : tensor<512x256x256xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<64x8x256xf32>, %arg1: tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<512x256x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x8x256xf32>
    %4 = stablehlo.maximum %3, %arg0 : tensor<64x8x256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x8x256x256xf32>
    %7 = stablehlo.exponential %6 : tensor<64x8x256x256xf32>
    return %7 : tensor<64x8x256x256xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x256x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    return %2 : tensor<64x8x256x256xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,32,256]{2,1,3,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
    %4 = mhlo.copy %3 : tensor<64x8x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x32x256xf32>) -> tensor<512x32x256xf32>
    return %5 : tensor<512x32x256xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<16384x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x8x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
    return %5 : tensor<512x256x32xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64x256xf32>, %arg7: tensor<64x256xf32>, %arg8: tensor<64x16x16x256xf32>, %arg9: tensor<256xf32>) -> tensor<16384x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x16x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x256x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %6 = mhlo.copy %5 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x256xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x256xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x256x256xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x256x256xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x256x256xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x256x256xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
    return %26 : tensor<16384x256xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<64x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x256xf32>, %arg5: tensor<64x16x16x256xf32>, %arg6: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
    %1 = stablehlo.add %arg5, %0 : tensor<64x16x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x256x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %6 = mhlo.copy %5 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg4, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x256xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x256xf32>
    %17 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x256x256xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<64x256x256xf32>
    return %20 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x256xf32>, %arg5: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x256x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %6 = mhlo.copy %5 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x256xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x256xf32>
    return %16 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x256x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %6 = mhlo.copy %5 : tensor<64x256x256xf32>
    %cst = stablehlo.constant dense<3.906250e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x256xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x256xf32>
    return %11 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<64x16x16x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x256x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %6 = mhlo.copy %5 : tensor<64x256x256xf32>
    return %6 : tensor<64x256x256xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,160,256]{1,0,2,3}"} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x160x256xf32>
    return %1 : tensor<3x3x160x256xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024x160xf32>, %arg4: tensor<64x1024xf32>) -> tensor<64x32x32x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024x160xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<64x32x32x160xf32>
    return %10 : tensor<64x32x32x160xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x1024xf32>
    return %4 : tensor<64x1024xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x1024x160xf32>
    return %4 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<65536x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x1024x160xf32>
    return %3 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<64x32x32x640xf32>, %arg1: tensor<640xf32>) -> tensor<65536x640xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<640xf32>) -> tensor<64x32x32x640xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x32x32x640xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x32x32x640xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
    %5 = mhlo.copy %4 : tensor<64x640x32x32xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
    %8 = mhlo.copy %7 : tensor<64x1024x640xf32>
    %9 = stablehlo.negate %2 : tensor<64x32x32x640xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x32x32x640xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
    %13 = mhlo.copy %12 : tensor<64x640x32x32xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
    %16 = mhlo.copy %15 : tensor<64x1024x640xf32>
    %17 = stablehlo.abs %16 : tensor<64x1024x640xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x1024x640xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x1024x640xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x1024x640xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x1024x640xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x1024x640xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x1024x640xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x1024x640xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x1024x640xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x1024x640xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x1024x640xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x1024x640xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x1024x640xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x1024x640xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x1024x640xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x1024x640xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %45 = stablehlo.negate %21 : tensor<64x1024x640xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %48 = stablehlo.exponential %45 : tensor<64x1024x640xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x1024x640xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x1024x640xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x1024x640xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x1024x640xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x1024x640xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x1024x640xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x1024x640xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x1024x640xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x1024x640xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x1024x640xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x1024x640xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x1024x640xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x1024x640xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x1024x640xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x1024x640xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x1024x640xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x1024x640xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x1024x640xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x1024x640xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x1024x640xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x1024x640xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x1024x640xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x1024x640xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x1024x640xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x1024x640xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x1024x640xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x1024x640xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x1024x640xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x1024x640xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x1024x640xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x1024x640xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x1024x640xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x1024x640xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x1024x640xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x1024x640xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x1024x640xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
    return %107 : tensor<65536x640xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x640xf32>
    return %1 : tensor<3x3x1x640xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<65536x640xf32>, %arg1: tensor<640xf32>) -> tensor<64x32x32x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,640,1024]{1,2,0}"} : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
    %4 = mhlo.copy %3 : tensor<64x640x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x640x1024xf32>) -> tensor<64x640x32x32xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,640]{2,1,3,0}"} : (tensor<64x640x32x32xf32>) -> tensor<64x32x32x640xf32>
    %7 = mhlo.copy %6 : tensor<64x32x32x640xf32>
    return %7 : tensor<64x32x32x640xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024x160xf32>, %arg4: tensor<64x1024xf32>) -> tensor<65536x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024x160xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    return %10 : tensor<65536x160xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x1024xf32>
    return %4 : tensor<64x1024xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x1024x160xf32>
    return %4 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<65536x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x1024x160xf32>
    return %3 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<320x1024x32xf32>) -> tensor<65536x160xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,5,32]{3,1,2,0}"} : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
    %2 = mhlo.copy %1 : tensor<64x1024x5x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x5x32xf32>) -> tensor<65536x160xf32>
    return %3 : tensor<65536x160xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<16384x160xf32>, %arg1: tensor<160xf32>) -> tensor<320x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,256,32]{3,1,2,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x5x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x256x32xf32>) -> tensor<320x256x32xf32>
    return %5 : tensor<320x256x32xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<64x5x1024x256xf32>, %arg1: tensor<64x5x1024xf32>) -> tensor<320x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<64x5x1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x256xf32>) -> tensor<320x1024x256xf32>
    return %2 : tensor<320x1024x256xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<64x5x1024xf32>, %arg1: tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<320x1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x5x1024xf32>
    %4 = stablehlo.maximum %3, %arg0 : tensor<64x5x1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x5x1024x256xf32>
    %7 = stablehlo.exponential %6 : tensor<64x5x1024x256xf32>
    return %7 : tensor<64x5x1024x256xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<320x1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    return %2 : tensor<64x5x1024x256xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<16384x160xf32>, %arg1: tensor<160xf32>) -> tensor<320x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,32,256]{2,1,3,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
    %4 = mhlo.copy %3 : tensor<64x5x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x32x256xf32>) -> tensor<320x32x256xf32>
    return %5 : tensor<320x32x256xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x160xf32>, %arg5: tensor<160xf32>) -> tensor<16384x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x160x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %6 = mhlo.copy %5 : tensor<64x256x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x160xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x160xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x160xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x160xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
    return %17 : tensor<16384x160xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x256x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x160x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %6 = mhlo.copy %5 : tensor<64x256x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x160xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x160xf32>
    return %11 : tensor<64x256x160xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<64x16x16x160xf32>, %arg1: tensor<160xf32>) -> tensor<64x256x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x160x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %6 = mhlo.copy %5 : tensor<64x256x160xf32>
    return %6 : tensor<64x256x160xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %1 = mhlo.copy %0 : tensor<2x2x160x160xf32>
    return %1 : tensor<2x2x160x160xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024x160xf32>, %arg4: tensor<64x1024xf32>) -> tensor<64x32x32x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024x160xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,160,1024]{1,2,0}"} : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
    %11 = mhlo.copy %10 : tensor<64x160x1024xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
    %13 = stablehlo.transpose %12, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,160]{2,1,3,0}"} : (tensor<64x160x32x32xf32>) -> tensor<64x32x32x160xf32>
    %14 = mhlo.copy %13 : tensor<64x32x32x160xf32>
    return %14 : tensor<64x32x32x160xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<320x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,1024,32]{3,1,2,0}"} : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
    %4 = mhlo.copy %3 : tensor<64x5x1024x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x32xf32>) -> tensor<320x1024x32xf32>
    return %5 : tensor<320x1024x32xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024x160xf32>, %arg4: tensor<64x1024xf32>) -> tensor<65536x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024x160xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    return %10 : tensor<65536x160xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x1024xf32>
    return %4 : tensor<64x1024xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x1024x160xf32>
    return %4 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<65536x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x1024x160xf32>
    return %3 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<64x32x32x640xf32>, %arg1: tensor<640xf32>) -> tensor<65536x640xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<640xf32>) -> tensor<64x32x32x640xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x32x32x640xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x32x32x640xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
    %5 = mhlo.copy %4 : tensor<64x640x32x32xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
    %8 = mhlo.copy %7 : tensor<64x1024x640xf32>
    %9 = stablehlo.negate %2 : tensor<64x32x32x640xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x32x32x640xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
    %13 = mhlo.copy %12 : tensor<64x640x32x32xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
    %16 = mhlo.copy %15 : tensor<64x1024x640xf32>
    %17 = stablehlo.abs %16 : tensor<64x1024x640xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x1024x640xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x1024x640xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x1024x640xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x1024x640xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x1024x640xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x1024x640xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x1024x640xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x1024x640xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x1024x640xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x1024x640xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x1024x640xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x1024x640xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x1024x640xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x1024x640xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x1024x640xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %45 = stablehlo.negate %21 : tensor<64x1024x640xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %48 = stablehlo.exponential %45 : tensor<64x1024x640xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x1024x640xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x1024x640xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x1024x640xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x1024x640xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x1024x640xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x1024x640xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x1024x640xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x1024x640xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x1024x640xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x1024x640xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x1024x640xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x1024x640xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x1024x640xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x1024x640xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x1024x640xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x1024x640xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x1024x640xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x1024x640xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x1024x640xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x1024x640xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x1024x640xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x1024x640xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x1024x640xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x1024x640xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x1024x640xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x1024x640xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x1024x640xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x1024x640xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x1024x640xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x1024x640xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x1024x640xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x1024x640xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x1024x640xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x1024x640xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x1024x640xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x1024x640xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
    return %107 : tensor<65536x640xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x640xf32>
    return %1 : tensor<3x3x1x640xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<65536x640xf32>, %arg1: tensor<640xf32>) -> tensor<64x32x32x640xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x640xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,640,1024]{1,2,0}"} : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
    %4 = mhlo.copy %3 : tensor<64x640x1024xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x640x1024xf32>) -> tensor<64x640x32x32xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,640]{2,1,3,0}"} : (tensor<64x640x32x32xf32>) -> tensor<64x32x32x640xf32>
    %7 = mhlo.copy %6 : tensor<64x32x32x640xf32>
    return %7 : tensor<64x32x32x640xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024x160xf32>, %arg4: tensor<64x1024xf32>) -> tensor<65536x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x1024x160xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x1024x160xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x1024x160xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    return %10 : tensor<65536x160xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x1024xf32>
    return %4 : tensor<64x1024xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<64x1024xf32>) -> tensor<64x1024x160xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x1024x160xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x1024x160xf32>
    return %4 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<64x1024xf32>, %arg4: tensor<64x1024xf32>, %arg5: tensor<64x32x32x160xf32>, %arg6: tensor<160xf32>, %arg7: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<64x32x32x160xf32>
    %5 = stablehlo.transpose %4, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %6 = mhlo.copy %5 : tensor<64x160x32x32xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %8 = stablehlo.transpose %7, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %9 = mhlo.copy %8 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %11 = stablehlo.multiply %arg4, %10 : tensor<64x1024xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %13 = stablehlo.subtract %9, %12 : tensor<64x1024x160xf32>
    %14 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<64x1024x160xf32>
    %16 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x1024x160xf32>
    %18 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x1024x160xf32>
    %20 = stablehlo.add %2, %19 : tensor<64x1024x160xf32>
    return %20 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<320x1024x32xf32>) -> tensor<65536x160xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,5,32]{3,1,2,0}"} : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
    %2 = mhlo.copy %1 : tensor<64x1024x5x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x5x32xf32>) -> tensor<65536x160xf32>
    return %3 : tensor<65536x160xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<16384x160xf32>, %arg1: tensor<160xf32>) -> tensor<320x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,256,32]{3,1,2,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x5x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x256x32xf32>) -> tensor<320x256x32xf32>
    return %5 : tensor<320x256x32xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<64x5x1024x256xf32>, %arg1: tensor<64x5x1024xf32>) -> tensor<320x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<64x5x1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x256xf32>) -> tensor<320x1024x256xf32>
    return %2 : tensor<320x1024x256xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<64x5x1024xf32>, %arg1: tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<320x1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x5x1024xf32>
    %4 = stablehlo.maximum %3, %arg0 : tensor<64x5x1024xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x5x1024x256xf32>
    %7 = stablehlo.exponential %6 : tensor<64x5x1024x256xf32>
    return %7 : tensor<64x5x1024x256xf32>
  }
  func.func private @fused_computation.173(%arg0: tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<320x1024x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    return %2 : tensor<64x5x1024x256xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<16384x160xf32>, %arg1: tensor<160xf32>) -> tensor<320x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,32,256]{2,1,3,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
    %4 = mhlo.copy %3 : tensor<64x5x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x32x256xf32>) -> tensor<320x32x256xf32>
    return %5 : tensor<320x32x256xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x160xf32>, %arg5: tensor<160xf32>) -> tensor<16384x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x160x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %6 = mhlo.copy %5 : tensor<64x256x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x160xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x160xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x160xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x160xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
    return %17 : tensor<16384x160xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x256x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x160x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %6 = mhlo.copy %5 : tensor<64x256x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x160xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x160xf32>
    return %11 : tensor<64x256x160xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<64x16x16x160xf32>, %arg1: tensor<160xf32>) -> tensor<64x256x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x160x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
    %6 = mhlo.copy %5 : tensor<64x256x160xf32>
    return %6 : tensor<64x256x160xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %1 = mhlo.copy %0 : tensor<2x2x160x160xf32>
    return %1 : tensor<2x2x160x160xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024xf32>, %arg4: tensor<160xf32>, %arg5: tensor<160xf32>, %arg6: tensor<64x1024xf32>, %arg7: tensor<64x1024xf32>, %arg8: tensor<64x32x32x160xf32>, %arg9: tensor<160xf32>) -> tensor<64x32x32x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x32x32x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %3 = mhlo.copy %2 : tensor<64x160x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.copy %5 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x1024x160xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x1024x160xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024x160xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024x160xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x1024xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x1024x160xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x1024x160xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x1024x160xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x1024x160xf32>
    %26 = stablehlo.transpose %25, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,160,1024]{1,2,0}"} : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
    %27 = mhlo.copy %26 : tensor<64x160x1024xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
    %29 = stablehlo.transpose %28, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,160]{2,1,3,0}"} : (tensor<64x160x32x32xf32>) -> tensor<64x32x32x160xf32>
    %30 = mhlo.copy %29 : tensor<64x32x32x160xf32>
    return %30 : tensor<64x32x32x160xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<65536x160xf32>, %arg1: tensor<160xf32>) -> tensor<320x1024x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<65536x160xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,1024,32]{3,1,2,0}"} : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
    %4 = mhlo.copy %3 : tensor<64x5x1024x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x32xf32>) -> tensor<320x1024x32xf32>
    return %5 : tensor<320x1024x32xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024xf32>, %arg4: tensor<160xf32>, %arg5: tensor<160xf32>, %arg6: tensor<64x1024xf32>, %arg7: tensor<64x1024xf32>, %arg8: tensor<64x32x32x160xf32>, %arg9: tensor<160xf32>) -> tensor<65536x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x32x32x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %3 = mhlo.copy %2 : tensor<64x160x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.copy %5 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x1024x160xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x1024x160xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024x160xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024x160xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x1024xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x1024x160xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x1024x160xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x1024x160xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x1024x160xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
    return %26 : tensor<65536x160xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x1024xf32>
    return %4 : tensor<64x1024xf32>
  }
  func.func private @fused_computation.184(%arg0: tensor<64x1024xf32>, %arg1: tensor<160xf32>, %arg2: tensor<160xf32>, %arg3: tensor<64x1024xf32>, %arg4: tensor<64x1024xf32>, %arg5: tensor<64x32x32x160xf32>, %arg6: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %1 = stablehlo.add %arg5, %0 : tensor<64x32x32x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %3 = mhlo.copy %2 : tensor<64x160x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.copy %5 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %8 = stablehlo.multiply %arg4, %7 : tensor<64x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x1024x160xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x1024x160xf32>
    %13 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024x160xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024x160xf32>
    %17 = stablehlo.multiply %arg0, %7 : tensor<64x1024xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x1024x160xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<64x1024x160xf32>
    return %20 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<160xf32>, %arg1: tensor<160xf32>, %arg2: tensor<64x1024xf32>, %arg3: tensor<64x1024xf32>, %arg4: tensor<64x32x32x160xf32>, %arg5: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x32x32x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %3 = mhlo.copy %2 : tensor<64x160x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.copy %5 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x1024x160xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x1024x160xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x1024x160xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x1024x160xf32>
    return %16 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<64x1024xf32>) -> tensor<64x1024xf32> {
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x1024xf32>
    return %4 : tensor<64x1024xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<64x1024xf32>, %arg1: tensor<64x32x32x160xf32>, %arg2: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x32x32x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %3 = mhlo.copy %2 : tensor<64x160x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.copy %5 : tensor<64x1024x160xf32>
    %cst = stablehlo.constant dense<6.250000e-03> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x1024xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x1024x160xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x1024x160xf32>
    return %11 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<64x32x32x160xf32>, %arg1: tensor<160xf32>) -> tensor<64x1024x160xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x32x32x160xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
    %3 = mhlo.copy %2 : tensor<64x160x32x32xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
    %6 = mhlo.copy %5 : tensor<64x1024x160xf32>
    return %6 : tensor<64x1024x160xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,160]{1,0,2,3}"} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
    %1 = mhlo.copy %0 : tensor<3x3x64x160xf32>
    return %1 : tensor<3x3x64x160xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096x64xf32>, %arg4: tensor<64x4096xf32>) -> tensor<64x64x64x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<64x64x64x64xf32>
    return %10 : tensor<64x64x64x64xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<64x4096xf32>) -> tensor<64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x4096xf32>
    return %4 : tensor<64x4096xf32>
  }
  func.func private @fused_computation.192(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<64x4096xf32>) -> tensor<64x4096x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x4096x64xf32>
    return %4 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<262144x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x4096x64xf32>
    return %3 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<64x64x64x256xf32>, %arg1: tensor<256xf32>) -> tensor<262144x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<64x64x64x256xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x64x64x256xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x64x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
    %5 = mhlo.copy %4 : tensor<64x256x64x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
    %8 = mhlo.copy %7 : tensor<64x4096x256xf32>
    %9 = stablehlo.negate %2 : tensor<64x64x64x256xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x64x64x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
    %13 = mhlo.copy %12 : tensor<64x256x64x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
    %16 = mhlo.copy %15 : tensor<64x4096x256xf32>
    %17 = stablehlo.abs %16 : tensor<64x4096x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x4096x256xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x4096x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x4096x256xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x4096x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x4096x256xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x4096x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x4096x256xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x4096x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x4096x256xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x4096x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x4096x256xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x4096x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x4096x256xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x4096x256xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x4096x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %45 = stablehlo.negate %21 : tensor<64x4096x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %48 = stablehlo.exponential %45 : tensor<64x4096x256xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x4096x256xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x4096x256xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x4096x256xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x4096x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x4096x256xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x4096x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x4096x256xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x4096x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x4096x256xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x4096x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x4096x256xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x4096x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x4096x256xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x4096x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x4096x256xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x4096x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x4096x256xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x4096x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x4096x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x4096x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x4096x256xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x4096x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x4096x256xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x4096x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x4096x256xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x4096x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x4096x256xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x4096x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x4096x256xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x4096x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x4096x256xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x4096x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x4096x256xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x4096x256xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x4096x256xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x4096x256xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
    return %107 : tensor<262144x256xf32>
  }
  func.func private @fused_computation.195(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<262144x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,4096]{1,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
    %4 = mhlo.copy %3 : tensor<64x256x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,256]{2,1,3,0}"} : (tensor<64x256x64x64xf32>) -> tensor<64x64x64x256xf32>
    %7 = mhlo.copy %6 : tensor<64x64x64x256xf32>
    return %7 : tensor<64x64x64x256xf32>
  }
  func.func private @fused_computation.197(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096x64xf32>, %arg4: tensor<64x4096xf32>) -> tensor<262144x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    return %10 : tensor<262144x64xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<64x4096xf32>) -> tensor<64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x4096xf32>
    return %4 : tensor<64x4096xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<64x4096xf32>) -> tensor<64x4096x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x4096x64xf32>
    return %4 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.200(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<262144x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x4096x64xf32>
    return %3 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<128x4096x32xf32>) -> tensor<262144x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,4096,2,32]{3,1,2,0}"} : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
    %2 = mhlo.copy %1 : tensor<64x4096x2x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x4096x2x32xf32>) -> tensor<262144x64xf32>
    return %3 : tensor<262144x64xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<16384x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,256,32]{3,1,2,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x2x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x256x32xf32>) -> tensor<128x256x32xf32>
    return %5 : tensor<128x256x32xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<64x2x4096x256xf32>, %arg1: tensor<64x2x4096xf32>) -> tensor<128x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<64x2x4096x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x256xf32>) -> tensor<128x4096x256xf32>
    return %2 : tensor<128x4096x256xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<64x2x4096xf32>, %arg1: tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x4096x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x2x4096xf32>
    %4 = stablehlo.maximum %3, %arg0 : tensor<64x2x4096xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x2x4096x256xf32>
    %7 = stablehlo.exponential %6 : tensor<64x2x4096x256xf32>
    return %7 : tensor<64x2x4096x256xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x4096x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    return %2 : tensor<64x2x4096x256xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<16384x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,32,256]{2,1,3,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
    %4 = mhlo.copy %3 : tensor<64x2x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x32x256xf32>) -> tensor<128x32x256xf32>
    return %5 : tensor<128x32x256xf32>
  }
  func.func private @fused_computation.207(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x64xf32>, %arg5: tensor<64xf32>) -> tensor<16384x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x64x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %6 = mhlo.copy %5 : tensor<64x256x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x64xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
    return %17 : tensor<16384x64xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x256x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x64x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %6 = mhlo.copy %5 : tensor<64x256x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x64xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x64xf32>
    return %11 : tensor<64x256x64xf32>
  }
  func.func private @fused_computation.210(%arg0: tensor<64x16x16x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x256x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x64x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %6 = mhlo.copy %5 : tensor<64x256x64xf32>
    return %6 : tensor<64x256x64xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %1 = mhlo.copy %0 : tensor<4x4x64x64xf32>
    return %1 : tensor<4x4x64x64xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096x64xf32>, %arg4: tensor<64x4096xf32>) -> tensor<64x64x64x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,64,4096]{1,2,0}"} : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
    %11 = mhlo.copy %10 : tensor<64x64x4096xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
    %13 = stablehlo.transpose %12, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{2,1,3,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %14 = mhlo.copy %13 : tensor<64x64x64x64xf32>
    return %14 : tensor<64x64x64x64xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,4096,32]{3,1,2,0}"} : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
    %4 = mhlo.copy %3 : tensor<64x2x4096x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x32xf32>) -> tensor<128x4096x32xf32>
    return %5 : tensor<128x4096x32xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096x64xf32>, %arg4: tensor<64x4096xf32>) -> tensor<262144x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    return %10 : tensor<262144x64xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<64x4096xf32>) -> tensor<64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x4096xf32>
    return %4 : tensor<64x4096xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<64x4096xf32>) -> tensor<64x4096x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x4096x64xf32>
    return %4 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<262144x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x4096x64xf32>
    return %3 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<64x64x64x256xf32>, %arg1: tensor<256xf32>) -> tensor<262144x256xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<64x64x64x256xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x64x64x256xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x64x64x256xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
    %5 = mhlo.copy %4 : tensor<64x256x64x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
    %8 = mhlo.copy %7 : tensor<64x4096x256xf32>
    %9 = stablehlo.negate %2 : tensor<64x64x64x256xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x64x64x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
    %13 = mhlo.copy %12 : tensor<64x256x64x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
    %16 = mhlo.copy %15 : tensor<64x4096x256xf32>
    %17 = stablehlo.abs %16 : tensor<64x4096x256xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x4096x256xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x4096x256xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x4096x256xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x4096x256xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x4096x256xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x4096x256xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x4096x256xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x4096x256xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x4096x256xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x4096x256xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x4096x256xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x4096x256xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x4096x256xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x4096x256xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x4096x256xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %45 = stablehlo.negate %21 : tensor<64x4096x256xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %48 = stablehlo.exponential %45 : tensor<64x4096x256xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x4096x256xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x4096x256xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x4096x256xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x4096x256xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x4096x256xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x4096x256xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x4096x256xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x4096x256xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x4096x256xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x4096x256xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x4096x256xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x4096x256xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x4096x256xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x4096x256xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x4096x256xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x4096x256xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x4096x256xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x4096x256xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x4096x256xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x4096x256xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x4096x256xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x4096x256xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x4096x256xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x4096x256xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x4096x256xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x4096x256xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x4096x256xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x4096x256xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x4096x256xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x4096x256xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x4096x256xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x4096x256xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x4096x256xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x4096x256xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x4096x256xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x4096x256xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
    return %107 : tensor<262144x256xf32>
  }
  func.func private @fused_computation.219(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.220(%arg0: tensor<262144x256xf32>, %arg1: tensor<256xf32>) -> tensor<64x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,4096]{1,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
    %4 = mhlo.copy %3 : tensor<64x256x4096xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,256]{2,1,3,0}"} : (tensor<64x256x64x64xf32>) -> tensor<64x64x64x256xf32>
    %7 = mhlo.copy %6 : tensor<64x64x64x256xf32>
    return %7 : tensor<64x64x64x256xf32>
  }
  func.func private @fused_computation.221(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096x64xf32>, %arg4: tensor<64x4096xf32>) -> tensor<262144x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x4096x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x4096x64xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x4096x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    return %10 : tensor<262144x64xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<64x4096xf32>) -> tensor<64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x4096xf32>
    return %4 : tensor<64x4096xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<64x4096xf32>) -> tensor<64x4096x64xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x4096xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x4096x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x4096x64xf32>
    return %4 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x4096xf32>, %arg4: tensor<64x4096xf32>, %arg5: tensor<64x64x64x64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<64x64x64x64xf32>
    %5 = stablehlo.transpose %4, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %6 = mhlo.copy %5 : tensor<64x64x64x64xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %8 = stablehlo.transpose %7, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %9 = mhlo.copy %8 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %11 = stablehlo.multiply %arg4, %10 : tensor<64x4096xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %13 = stablehlo.subtract %9, %12 : tensor<64x4096x64xf32>
    %14 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<64x4096x64xf32>
    %16 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x4096x64xf32>
    %18 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x4096x64xf32>
    %20 = stablehlo.add %2, %19 : tensor<64x4096x64xf32>
    return %20 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.225(%arg0: tensor<128x4096x32xf32>) -> tensor<262144x64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,4096,2,32]{3,1,2,0}"} : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
    %2 = mhlo.copy %1 : tensor<64x4096x2x32xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x4096x2x32xf32>) -> tensor<262144x64xf32>
    return %3 : tensor<262144x64xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<16384x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,256,32]{3,1,2,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
    %4 = mhlo.copy %3 : tensor<64x2x256x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x256x32xf32>) -> tensor<128x256x32xf32>
    return %5 : tensor<128x256x32xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<64x2x4096x256xf32>, %arg1: tensor<64x2x4096xf32>) -> tensor<128x4096x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %1 = stablehlo.divide %arg0, %0 : tensor<64x2x4096x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x256xf32>) -> tensor<128x4096x256xf32>
    return %2 : tensor<128x4096x256xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<64x2x4096xf32>, %arg1: tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<128x4096x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x2x4096xf32>
    %4 = stablehlo.maximum %3, %arg0 : tensor<64x2x4096xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x2x4096x256xf32>
    %7 = stablehlo.exponential %6 : tensor<64x2x4096x256xf32>
    return %7 : tensor<64x2x4096x256xf32>
  }
  func.func private @fused_computation.229(%arg0: tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x4096x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    return %2 : tensor<64x2x4096x256xf32>
  }
  func.func private @fused_computation.230(%arg0: tensor<16384x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,32,256]{2,1,3,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
    %4 = mhlo.copy %3 : tensor<64x2x32x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x32x256xf32>) -> tensor<128x32x256xf32>
    return %5 : tensor<128x32x256xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x64xf32>, %arg5: tensor<64xf32>) -> tensor<16384x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x64x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %6 = mhlo.copy %5 : tensor<64x256x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x64xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
    return %17 : tensor<16384x64xf32>
  }
  func.func private @fused_computation.232(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x256x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x64x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %6 = mhlo.copy %5 : tensor<64x256x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x64xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x64xf32>
    return %11 : tensor<64x256x64xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<64x16x16x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x256x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x64x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
    %6 = mhlo.copy %5 : tensor<64x256x64xf32>
    return %6 : tensor<64x256x64xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %1 = mhlo.copy %0 : tensor<4x4x64x64xf32>
    return %1 : tensor<4x4x64x64xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x4096xf32>, %arg7: tensor<64x4096xf32>, %arg8: tensor<64x64x64x64xf32>, %arg9: tensor<64xf32>) -> tensor<64x64x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x64x64x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %3 = mhlo.copy %2 : tensor<64x64x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.copy %5 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x4096x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x4096x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x4096x64xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x4096x64xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x4096x64xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x4096x64xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x4096x64xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x4096x64xf32>
    %26 = stablehlo.transpose %25, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,64,4096]{1,2,0}"} : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
    %27 = mhlo.copy %26 : tensor<64x64x4096xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
    %29 = stablehlo.transpose %28, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{2,1,3,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %30 = mhlo.copy %29 : tensor<64x64x64x64xf32>
    return %30 : tensor<64x64x64x64xf32>
  }
  func.func private @fused_computation.237(%arg0: tensor<262144x64xf32>, %arg1: tensor<64xf32>) -> tensor<128x4096x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<262144x64xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,4096,32]{3,1,2,0}"} : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
    %4 = mhlo.copy %3 : tensor<64x2x4096x32xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x32xf32>) -> tensor<128x4096x32xf32>
    return %5 : tensor<128x4096x32xf32>
  }
  func.func private @fused_computation.238(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x4096xf32>, %arg7: tensor<64x4096xf32>, %arg8: tensor<64x64x64x64xf32>, %arg9: tensor<64xf32>) -> tensor<262144x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x64x64x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %3 = mhlo.copy %2 : tensor<64x64x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.copy %5 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x4096x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x4096x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x4096x64xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x4096x64xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x4096x64xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x4096x64xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x4096x64xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x4096x64xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
    return %26 : tensor<262144x64xf32>
  }
  func.func private @fused_computation.239(%arg0: tensor<64x4096xf32>) -> tensor<64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x4096xf32>
    return %4 : tensor<64x4096xf32>
  }
  func.func private @fused_computation.240(%arg0: tensor<64x4096xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x4096xf32>, %arg4: tensor<64x4096xf32>, %arg5: tensor<64x64x64x64xf32>, %arg6: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %1 = stablehlo.add %arg5, %0 : tensor<64x64x64x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %3 = mhlo.copy %2 : tensor<64x64x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.copy %5 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %8 = stablehlo.multiply %arg4, %7 : tensor<64x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x4096x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x4096x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x4096x64xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x4096x64xf32>
    %17 = stablehlo.multiply %arg0, %7 : tensor<64x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x4096x64xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<64x4096x64xf32>
    return %20 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.241(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x4096xf32>, %arg3: tensor<64x4096xf32>, %arg4: tensor<64x64x64x64xf32>, %arg5: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x64x64x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %3 = mhlo.copy %2 : tensor<64x64x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.copy %5 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x4096x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x4096x64xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x4096x64xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x4096x64xf32>
    return %16 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.242(%arg0: tensor<64x4096xf32>) -> tensor<64x4096xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x4096xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %3 = stablehlo.add %1, %2 : tensor<64x4096xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64x4096xf32>
    return %4 : tensor<64x4096xf32>
  }
  func.func private @fused_computation.243(%arg0: tensor<64x4096xf32>, %arg1: tensor<64x64x64x64xf32>, %arg2: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x64x64x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %3 = mhlo.copy %2 : tensor<64x64x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.copy %5 : tensor<64x4096x64xf32>
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x4096xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x4096x64xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x4096x64xf32>
    return %11 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.244(%arg0: tensor<64x64x64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x4096x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x64x64x64xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
    %3 = mhlo.copy %2 : tensor<64x64x64x64xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
    %6 = mhlo.copy %5 : tensor<64x4096x64xf32>
    return %6 : tensor<64x4096x64xf32>
  }
  func.func private @fused_computation.245(%arg0: tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x32x64xf32>
    return %1 : tensor<3x3x32x64xf32>
  }
  func.func private @fused_computation.246(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384x32xf32>, %arg4: tensor<64x16384xf32>) -> tensor<64x128x128x32xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384x32xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<64x128x128x32xf32>
    return %10 : tensor<64x128x128x32xf32>
  }
  func.func private @region_2.3.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.247(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<64x16384xf32>) -> tensor<64x16384xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %6 = stablehlo.multiply %5, %0 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.add %6, %7 : tensor<64x16384xf32>
    %9 = stablehlo.rsqrt %8 : tensor<64x16384xf32>
    return %9 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.248(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<1048576x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x16384x32xf32>
    return %3 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.249(%arg0: tensor<64x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<1048576x128xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x128x128x128xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
    %5 = mhlo.copy %4 : tensor<64x128x128x128xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
    %8 = mhlo.copy %7 : tensor<64x16384x128xf32>
    %9 = stablehlo.negate %2 : tensor<64x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x128x128x128xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
    %13 = mhlo.copy %12 : tensor<64x128x128x128xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
    %16 = mhlo.copy %15 : tensor<64x16384x128xf32>
    %17 = stablehlo.abs %16 : tensor<64x16384x128xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x16384x128xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x16384x128xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x16384x128xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x16384x128xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x16384x128xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x16384x128xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x16384x128xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x16384x128xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x16384x128xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x16384x128xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x16384x128xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x16384x128xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x16384x128xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x16384x128xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x16384x128xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %45 = stablehlo.negate %21 : tensor<64x16384x128xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %48 = stablehlo.exponential %45 : tensor<64x16384x128xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x16384x128xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x16384x128xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x16384x128xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x16384x128xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x16384x128xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x16384x128xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x16384x128xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x16384x128xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x16384x128xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x16384x128xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x16384x128xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x16384x128xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x16384x128xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x16384x128xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x16384x128xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x16384x128xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x16384x128xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x16384x128xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x16384x128xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x16384x128xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x16384x128xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x16384x128xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x16384x128xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x16384x128xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x16384x128xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x16384x128xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x16384x128xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x16384x128xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x16384x128xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x16384x128xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x16384x128xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x16384x128xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x16384x128xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x16384x128xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x16384x128xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x16384x128xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
    return %107 : tensor<1048576x128xf32>
  }
  func.func private @fused_computation.250(%arg0: tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x128xf32>
    return %1 : tensor<3x3x1x128xf32>
  }
  func.func private @fused_computation.251(%arg0: tensor<1048576x128xf32>, %arg1: tensor<128xf32>) -> tensor<64x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,128,16384]{1,2,0}"} : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
    %4 = mhlo.copy %3 : tensor<64x128x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x16384xf32>) -> tensor<64x128x128x128xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{2,1,3,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
    %7 = mhlo.copy %6 : tensor<64x128x128x128xf32>
    return %7 : tensor<64x128x128x128xf32>
  }
  func.func private @fused_computation.252(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384x32xf32>, %arg4: tensor<64x16384xf32>) -> tensor<1048576x32xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384x32xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    return %10 : tensor<1048576x32xf32>
  }
  func.func private @region_2.3.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.253(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<64x16384xf32>) -> tensor<64x16384xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %6 = stablehlo.multiply %5, %0 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.add %6, %7 : tensor<64x16384xf32>
    %9 = stablehlo.rsqrt %8 : tensor<64x16384xf32>
    return %9 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.254(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<1048576x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x16384x32xf32>
    return %3 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.255(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
    return %2 : tensor<64x256x32xf32>
  }
  func.func private @fused_computation.256(%arg0: tensor<64x1x16384x256xf32>, %arg1: tensor<64x1x16384xf32>) -> tensor<64x16384x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<64x1x16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x16384x256xf32>) -> tensor<64x16384x256xf32>
    return %3 : tensor<64x16384x256xf32>
  }
  func.func private @fused_computation.257(%arg0: tensor<64x1x16384xf32>, %arg1: tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %3 = mhlo.copy %2 : tensor<64x1x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1x16384xf32>
    %5 = stablehlo.maximum %4, %arg0 : tensor<64x1x16384xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x1x16384x256xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x16384x256xf32>
    return %9 : tensor<64x1x16384x256xf32>
  }
  func.func private @fused_computation.258(%arg0: tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %3 = mhlo.copy %2 : tensor<64x1x16384x256xf32>
    return %3 : tensor<64x1x16384x256xf32>
  }
  func.func private @fused_computation.259(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,1,256,32]{3,2,0,1}"} : (tensor<16384x32xf32>) -> tensor<64x1x256x32xf32>
    %3 = mhlo.copy %2 : tensor<64x1x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[64,1,32,256]{2,3,1,0}"} : (tensor<64x1x256x32xf32>) -> tensor<64x1x32x256xf32>
    %5 = mhlo.copy %4 : tensor<64x1x32x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x32x256xf32>) -> tensor<64x32x256xf32>
    return %6 : tensor<64x32x256xf32>
  }
  func.func private @fused_computation.260(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x32xf32>, %arg5: tensor<32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x32x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %6 = mhlo.copy %5 : tensor<64x256x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x32xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x32xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x32xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x32xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
    return %17 : tensor<16384x32xf32>
  }
  func.func private @region_6.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.261(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x32x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %6 = mhlo.copy %5 : tensor<64x256x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x32xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
    %13 = stablehlo.multiply %12, %7 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x256xf32>
    %16 = stablehlo.rsqrt %15 : tensor<64x256xf32>
    return %16 : tensor<64x256xf32>
  }
  func.func private @fused_computation.262(%arg0: tensor<64x16x16x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x32x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %6 = mhlo.copy %5 : tensor<64x256x32xf32>
    return %6 : tensor<64x256x32xf32>
  }
  func.func private @fused_computation.263(%arg0: tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %1 = mhlo.copy %0 : tensor<8x8x32x32xf32>
    return %1 : tensor<8x8x32x32xf32>
  }
  func.func private @fused_computation.264(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384x32xf32>, %arg4: tensor<64x16384xf32>) -> tensor<64x128x128x32xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384x32xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,32,16384]{1,2,0}"} : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
    %11 = mhlo.copy %10 : tensor<64x32x16384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
    %13 = stablehlo.transpose %12, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,32]{2,1,3,0}"} : (tensor<64x32x128x128xf32>) -> tensor<64x128x128x32xf32>
    %14 = mhlo.copy %13 : tensor<64x128x128x32xf32>
    return %14 : tensor<64x128x128x32xf32>
  }
  func.func private @fused_computation.265(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    return %2 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.266(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384x32xf32>, %arg4: tensor<64x16384xf32>) -> tensor<1048576x32xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384x32xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    return %10 : tensor<1048576x32xf32>
  }
  func.func private @region_2.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.267(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<64x16384xf32>) -> tensor<64x16384xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %6 = stablehlo.multiply %5, %0 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.add %6, %7 : tensor<64x16384xf32>
    %9 = stablehlo.rsqrt %8 : tensor<64x16384xf32>
    return %9 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.268(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<1048576x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<64x16384x32xf32>
    return %3 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.269(%arg0: tensor<64x128x128x128xf32>, %arg1: tensor<128xf32>) -> tensor<1048576x128xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<64x128x128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64x128x128x128xf32>
    %4 = stablehlo.transpose %3, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
    %5 = mhlo.copy %4 : tensor<64x128x128x128xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %7 = stablehlo.transpose %6, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
    %8 = mhlo.copy %7 : tensor<64x16384x128xf32>
    %9 = stablehlo.negate %2 : tensor<64x128x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x128x128x128xf32>
    %12 = stablehlo.transpose %11, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
    %13 = mhlo.copy %12 : tensor<64x128x128x128xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
    %16 = mhlo.copy %15 : tensor<64x16384x128xf32>
    %17 = stablehlo.abs %16 : tensor<64x16384x128xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %19 = stablehlo.compare LT, %17, %18 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %21 = stablehlo.multiply %16, %16 : tensor<64x16384x128xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<64x16384x128xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x16384x128xf32>
    %25 = stablehlo.multiply %24, %21 : tensor<64x16384x128xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %27 = stablehlo.add %25, %26 : tensor<64x16384x128xf32>
    %28 = stablehlo.multiply %27, %21 : tensor<64x16384x128xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %30 = stablehlo.add %28, %29 : tensor<64x16384x128xf32>
    %31 = stablehlo.multiply %30, %21 : tensor<64x16384x128xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %33 = stablehlo.add %31, %32 : tensor<64x16384x128xf32>
    %34 = stablehlo.multiply %33, %21 : tensor<64x16384x128xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %36 = stablehlo.add %34, %35 : tensor<64x16384x128xf32>
    %37 = stablehlo.multiply %36, %21 : tensor<64x16384x128xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %39 = stablehlo.add %37, %38 : tensor<64x16384x128xf32>
    %40 = stablehlo.multiply %16, %39 : tensor<64x16384x128xf32>
    %41 = stablehlo.subtract %18, %40 : tensor<64x16384x128xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %43 = stablehlo.compare LT, %16, %42 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %45 = stablehlo.negate %21 : tensor<64x16384x128xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %47 = stablehlo.compare LT, %45, %46 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %48 = stablehlo.exponential %45 : tensor<64x16384x128xf32>
    %49 = stablehlo.divide %18, %17 : tensor<64x16384x128xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<64x16384x128xf32>
    %51 = stablehlo.compare LT, %17, %44 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %53 = stablehlo.divide %18, %21 : tensor<64x16384x128xf32>
    %54 = stablehlo.multiply %52, %53 : tensor<64x16384x128xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %56 = stablehlo.add %54, %55 : tensor<64x16384x128xf32>
    %57 = stablehlo.multiply %56, %53 : tensor<64x16384x128xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %59 = stablehlo.add %57, %58 : tensor<64x16384x128xf32>
    %60 = stablehlo.multiply %59, %53 : tensor<64x16384x128xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %62 = stablehlo.add %60, %61 : tensor<64x16384x128xf32>
    %63 = stablehlo.multiply %62, %53 : tensor<64x16384x128xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %64 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %65 = stablehlo.add %63, %64 : tensor<64x16384x128xf32>
    %66 = stablehlo.multiply %65, %53 : tensor<64x16384x128xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %67 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %68 = stablehlo.add %66, %67 : tensor<64x16384x128xf32>
    %69 = stablehlo.multiply %68, %53 : tensor<64x16384x128xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %71 = stablehlo.add %69, %70 : tensor<64x16384x128xf32>
    %72 = stablehlo.multiply %71, %53 : tensor<64x16384x128xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %74 = stablehlo.add %72, %73 : tensor<64x16384x128xf32>
    %75 = stablehlo.multiply %74, %53 : tensor<64x16384x128xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %77 = stablehlo.add %75, %76 : tensor<64x16384x128xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %79 = stablehlo.multiply %78, %53 : tensor<64x16384x128xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %81 = stablehlo.add %79, %80 : tensor<64x16384x128xf32>
    %82 = stablehlo.multiply %81, %53 : tensor<64x16384x128xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %83 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %84 = stablehlo.add %82, %83 : tensor<64x16384x128xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<64x16384x128xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %87 = stablehlo.add %85, %86 : tensor<64x16384x128xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<64x16384x128xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %90 = stablehlo.add %88, %89 : tensor<64x16384x128xf32>
    %91 = stablehlo.multiply %90, %53 : tensor<64x16384x128xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %93 = stablehlo.add %91, %92 : tensor<64x16384x128xf32>
    %94 = stablehlo.multiply %93, %53 : tensor<64x16384x128xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %95 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %96 = stablehlo.add %94, %95 : tensor<64x16384x128xf32>
    %97 = stablehlo.multiply %96, %53 : tensor<64x16384x128xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
    %99 = stablehlo.add %97, %98 : tensor<64x16384x128xf32>
    %100 = stablehlo.select %51, %77, %99 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %101 = stablehlo.multiply %50, %100 : tensor<64x16384x128xf32>
    %102 = stablehlo.select %47, %42, %101 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %103 = stablehlo.subtract %44, %102 : tensor<64x16384x128xf32>
    %104 = stablehlo.select %43, %103, %102 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %105 = stablehlo.select %19, %41, %104 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
    %106 = stablehlo.multiply %8, %105 : tensor<64x16384x128xf32>
    %107 = mhlo.bitcast %106 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
    return %107 : tensor<1048576x128xf32>
  }
  func.func private @fused_computation.270(%arg0: tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x128xf32>
    return %1 : tensor<3x3x1x128xf32>
  }
  func.func private @fused_computation.271(%arg0: tensor<1048576x128xf32>, %arg1: tensor<128xf32>) -> tensor<64x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x128xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,128,16384]{1,2,0}"} : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
    %4 = mhlo.copy %3 : tensor<64x128x16384xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x16384xf32>) -> tensor<64x128x128x128xf32>
    %6 = stablehlo.transpose %5, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{2,1,3,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
    %7 = mhlo.copy %6 : tensor<64x128x128x128xf32>
    return %7 : tensor<64x128x128x128xf32>
  }
  func.func private @fused_computation.272(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384x32xf32>, %arg4: tensor<64x16384xf32>) -> tensor<1048576x32xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x16384x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<64x16384x32xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %9 = stablehlo.add %7, %8 : tensor<64x16384x32xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    return %10 : tensor<1048576x32xf32>
  }
  func.func private @region_2.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.273(%arg0: tensor<64x16384x32xf32>, %arg1: tensor<64x16384xf32>) -> tensor<64x16384xf32> {
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<64x16384x32xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %6 = stablehlo.multiply %5, %0 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.add %6, %7 : tensor<64x16384xf32>
    %9 = stablehlo.rsqrt %8 : tensor<64x16384xf32>
    return %9 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.274(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x16384xf32>, %arg4: tensor<64x16384xf32>, %arg5: tensor<64x128x128x32xf32>, %arg6: tensor<32xf32>, %arg7: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    %3 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %4 = stablehlo.add %arg5, %3 : tensor<64x128x128x32xf32>
    %5 = stablehlo.transpose %4, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %6 = mhlo.copy %5 : tensor<64x32x128x128xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %8 = stablehlo.transpose %7, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %9 = mhlo.copy %8 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %11 = stablehlo.multiply %arg4, %10 : tensor<64x16384xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %13 = stablehlo.subtract %9, %12 : tensor<64x16384x32xf32>
    %14 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<64x16384x32xf32>
    %16 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<64x16384x32xf32>
    %18 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x16384x32xf32>
    %20 = stablehlo.add %2, %19 : tensor<64x16384x32xf32>
    return %20 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.275(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
    return %2 : tensor<64x256x32xf32>
  }
  func.func private @fused_computation.276(%arg0: tensor<64x1x16384x256xf32>, %arg1: tensor<64x1x16384xf32>) -> tensor<64x16384x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<64x1x16384x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x16384x256xf32>) -> tensor<64x16384x256xf32>
    return %3 : tensor<64x16384x256xf32>
  }
  func.func private @fused_computation.277(%arg0: tensor<64x1x16384xf32>, %arg1: tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<64x16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %3 = mhlo.copy %2 : tensor<64x1x16384x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x1x16384xf32>
    %5 = stablehlo.maximum %4, %arg0 : tensor<64x1x16384xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
    %8 = stablehlo.subtract %3, %7 : tensor<64x1x16384x256xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x16384x256xf32>
    return %9 : tensor<64x1x16384x256xf32>
  }
  func.func private @fused_computation.278(%arg0: tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32> {
    %cst = stablehlo.constant dense<0.176776692> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x16384x256xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %3 = mhlo.copy %2 : tensor<64x1x16384x256xf32>
    return %3 : tensor<64x1x16384x256xf32>
  }
  func.func private @fused_computation.279(%arg0: tensor<16384x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16384x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,1,256,32]{3,2,0,1}"} : (tensor<16384x32xf32>) -> tensor<64x1x256x32xf32>
    %3 = mhlo.copy %2 : tensor<64x1x256x32xf32>
    %4 = stablehlo.transpose %3, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[64,1,32,256]{2,3,1,0}"} : (tensor<64x1x256x32xf32>) -> tensor<64x1x32x256xf32>
    %5 = mhlo.copy %4 : tensor<64x1x32x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x32x256xf32>) -> tensor<64x32x256xf32>
    return %6 : tensor<64x32x256xf32>
  }
  func.func private @fused_computation.280(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<64x16x16x32xf32>, %arg5: tensor<32xf32>) -> tensor<16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x16x16x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x32x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %6 = mhlo.copy %5 : tensor<64x256x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x32xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x256x32xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x256x32xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x256x32xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
    return %17 : tensor<16384x32xf32>
  }
  func.func private @region_6.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.281(%arg0: tensor<64x256xf32>, %arg1: tensor<64x16x16x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x16x16x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x32x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %6 = mhlo.copy %5 : tensor<64x256x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x256xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x256x32xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x256x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
    %13 = stablehlo.multiply %12, %7 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x256xf32>
    %16 = stablehlo.rsqrt %15 : tensor<64x256xf32>
    return %16 : tensor<64x256xf32>
  }
  func.func private @fused_computation.282(%arg0: tensor<64x16x16x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x256x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x16x16x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
    %3 = mhlo.copy %2 : tensor<64x32x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
    %6 = mhlo.copy %5 : tensor<64x256x32xf32>
    return %6 : tensor<64x256x32xf32>
  }
  func.func private @fused_computation.283(%arg0: tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %1 = mhlo.copy %0 : tensor<8x8x32x32xf32>
    return %1 : tensor<8x8x32x32xf32>
  }
  func.func private @fused_computation.284(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<64x16384xf32>, %arg7: tensor<64x16384xf32>, %arg8: tensor<64x128x128x32xf32>, %arg9: tensor<32xf32>) -> tensor<64x128x128x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x128x128x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x32x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.copy %5 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x16384x32xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x16384x32xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x16384x32xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x16384x32xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x16384xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x16384x32xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x16384x32xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x16384x32xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x16384x32xf32>
    %26 = stablehlo.transpose %25, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,32,16384]{1,2,0}"} : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
    %27 = mhlo.copy %26 : tensor<64x32x16384xf32>
    %28 = mhlo.bitcast %27 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
    %29 = stablehlo.transpose %28, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,32]{2,1,3,0}"} : (tensor<64x32x128x128xf32>) -> tensor<64x128x128x32xf32>
    %30 = mhlo.copy %29 : tensor<64x128x128x32xf32>
    return %30 : tensor<64x128x128x32xf32>
  }
  func.func private @fused_computation.285(%arg0: tensor<1048576x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1048576x32xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
    return %2 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.286(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32xf32>, %arg6: tensor<64x16384xf32>, %arg7: tensor<64x16384xf32>, %arg8: tensor<64x128x128x32xf32>, %arg9: tensor<32xf32>) -> tensor<1048576x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %1 = stablehlo.add %arg8, %0 : tensor<64x128x128x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x32x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.copy %5 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.multiply %arg7, %7 : tensor<64x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x16384x32xf32>
    %11 = stablehlo.broadcast_in_dim %arg6, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x16384x32xf32>
    %13 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x16384x32xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x16384x32xf32>
    %17 = stablehlo.multiply %arg3, %7 : tensor<64x16384xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x16384x32xf32>
    %20 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<64x16384x32xf32>
    %22 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %23 = stablehlo.multiply %21, %22 : tensor<64x16384x32xf32>
    %24 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %25 = stablehlo.add %23, %24 : tensor<64x16384x32xf32>
    %26 = mhlo.bitcast %25 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    return %26 : tensor<1048576x32xf32>
  }
  func.func private @region_2.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.287(%arg0: tensor<64x16384xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x16384xf32>, %arg4: tensor<64x16384xf32>, %arg5: tensor<64x128x128x32xf32>, %arg6: tensor<32xf32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %1 = stablehlo.add %arg5, %0 : tensor<64x128x128x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x32x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.copy %5 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.multiply %arg4, %7 : tensor<64x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x16384x32xf32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x16384x32xf32>
    %13 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x16384x32xf32>
    %15 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x16384x32xf32>
    %17 = stablehlo.multiply %arg0, %7 : tensor<64x16384xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %19 = stablehlo.subtract %16, %18 : tensor<64x16384x32xf32>
    %20 = stablehlo.multiply %19, %19 : tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %22 = stablehlo.multiply %21, %7 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %24 = stablehlo.add %22, %23 : tensor<64x16384xf32>
    %25 = stablehlo.rsqrt %24 : tensor<64x16384xf32>
    return %25 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.288(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64x16384xf32>, %arg3: tensor<64x16384xf32>, %arg4: tensor<64x128x128x32xf32>, %arg5: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<64x128x128x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x32x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.copy %5 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.multiply %arg3, %7 : tensor<64x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x16384x32xf32>
    %11 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<64x16384x32xf32>
    %13 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<64x16384x32xf32>
    %15 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
    %16 = stablehlo.add %14, %15 : tensor<64x16384x32xf32>
    return %16 : tensor<64x16384x32xf32>
  }
  func.func private @region_2.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.289(%arg0: tensor<64x16384xf32>, %arg1: tensor<64x128x128x32xf32>, %arg2: tensor<32xf32>) -> tensor<64x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<64x128x128x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x32x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.copy %5 : tensor<64x16384x32xf32>
    %cst = stablehlo.constant dense<3.125000e-02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<64x16384xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<64x16384x32xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %13 = stablehlo.multiply %12, %7 : tensor<64x16384xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
    %15 = stablehlo.add %13, %14 : tensor<64x16384xf32>
    %16 = stablehlo.rsqrt %15 : tensor<64x16384xf32>
    return %16 : tensor<64x16384xf32>
  }
  func.func private @fused_computation.290(%arg0: tensor<64x128x128x32xf32>, %arg1: tensor<32xf32>) -> tensor<64x16384x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64x128x128x32xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
    %3 = mhlo.copy %2 : tensor<64x32x128x128xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
    %5 = stablehlo.transpose %4, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
    %6 = mhlo.copy %5 : tensor<64x16384x32xf32>
    return %6 : tensor<64x16384x32xf32>
  }
  func.func private @fused_computation.291(%arg0: tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,32]{1,0,2,3}"} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
    %1 = mhlo.copy %0 : tensor<7x7x3x32xf32>
    return %1 : tensor<7x7x3x32xf32>
  }
  func.func private @fused_computation.292(%arg0: tensor<64x3x512x512xf32>) -> tensor<64x512x512x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,512,512,3]{2,1,3,0}"} : (tensor<64x3x512x512xf32>) -> tensor<64x512x512x3xf32>
    %1 = mhlo.copy %0 : tensor<64x512x512x3xf32>
    return %1 : tensor<64x512x512x3xf32>
  }
  func.func private @region_18.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    return %0 : tensor<64x4096x2xf32>
  }
  func.func private @region_18.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    return %0 : tensor<64x4096x2xf32>
  }
  func.func private @region_22.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<64x256x64xf32>, %arg1: tensor<f32>) -> tensor<64x256x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256x2xf32>
    return %0 : tensor<64x256x2xf32>
  }
  func.func private @region_23.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<64x2x4096x256xf32>, %arg1: tensor<f32>) -> tensor<64x2x4096x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096x8xf32>
    return %0 : tensor<64x2x4096x8xf32>
  }
  func.func private @region_18.23.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    return %0 : tensor<64x4096x2xf32>
  }
  func.func private @region_18.23.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    return %0 : tensor<64x4096x2xf32>
  }
  func.func private @region_22.29.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<64x256x64xf32>, %arg1: tensor<f32>) -> tensor<64x256x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256x2xf32>
    return %0 : tensor<64x256x2xf32>
  }
  func.func private @region_28.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<64x2x4096x256xf32>, %arg1: tensor<f32>) -> tensor<64x2x4096x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096x8xf32>
    return %0 : tensor<64x2x4096x8xf32>
  }
  func.func private @region_18.23.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    return %0 : tensor<64x4096x2xf32>
  }
  func.func private @region_18.23.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<64x4096x64xf32>, %arg1: tensor<f32>) -> tensor<64x4096x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    return %0 : tensor<64x4096x2xf32>
  }
  func.func private @region_34.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    return %0 : tensor<64x1024x5xf32>
  }
  func.func private @region_34.42.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    return %0 : tensor<64x1024x5xf32>
  }
  func.func private @region_38.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<64x256x160xf32>, %arg1: tensor<f32>) -> tensor<64x256x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256x5xf32>
    return %0 : tensor<64x256x5xf32>
  }
  func.func private @region_39.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<64x5x1024x256xf32>, %arg1: tensor<f32>) -> tensor<64x5x1024x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024x8xf32>
    return %0 : tensor<64x5x1024x8xf32>
  }
  func.func private @region_34.42.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    return %0 : tensor<64x1024x5xf32>
  }
  func.func private @region_34.42.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    return %0 : tensor<64x1024x5xf32>
  }
  func.func private @region_38.48.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<64x256x160xf32>, %arg1: tensor<f32>) -> tensor<64x256x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256x5xf32>
    return %0 : tensor<64x256x5xf32>
  }
  func.func private @region_44.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<64x5x1024x256xf32>, %arg1: tensor<f32>) -> tensor<64x5x1024x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024x8xf32>
    return %0 : tensor<64x5x1024x8xf32>
  }
  func.func private @region_34.42.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    return %0 : tensor<64x1024x5xf32>
  }
  func.func private @region_34.42.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<64x1024x160xf32>, %arg1: tensor<f32>) -> tensor<64x1024x5xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    return %0 : tensor<64x1024x5xf32>
  }
  func.func private @region_50.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    return %0 : tensor<64x256x8xf32>
  }
  func.func private @region_50.61.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    return %0 : tensor<64x256x8xf32>
  }
  func.func private @region_52.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<64x8x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x8x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256x8xf32>
    return %0 : tensor<64x8x256x8xf32>
  }
  func.func private @region_50.61.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    return %0 : tensor<64x256x8xf32>
  }
  func.func private @region_50.61.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    return %0 : tensor<64x256x8xf32>
  }
  func.func private @region_56.68(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<64x8x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x8x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256x8xf32>
    return %0 : tensor<64x8x256x8xf32>
  }
  func.func private @region_50.61.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    return %0 : tensor<64x256x8xf32>
  }
  func.func private @region_50.61.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<64x256x256xf32>, %arg1: tensor<f32>) -> tensor<64x256x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    return %0 : tensor<64x256x8xf32>
  }
  func.func @main(%arg0: tensor<32x3x7x7xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<64x32x3x3xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<160x64x3x3xf32>, %arg9: tensor<160xf32>, %arg10: tensor<160xf32>, %arg11: tensor<160xf32>, %arg12: tensor<256x160x3x3xf32>, %arg13: tensor<256xf32>, %arg14: tensor<256xf32>, %arg15: tensor<256xf32>, %arg16: tensor<32xf32>, %arg17: tensor<32xf32>, %arg18: tensor<32x32xf32>, %arg19: tensor<32xf32>, %arg20: tensor<32x32xf32>, %arg21: tensor<32xf32>, %arg22: tensor<32x32xf32>, %arg23: tensor<32xf32>, %arg24: tensor<32x32x8x8xf32>, %arg25: tensor<32xf32>, %arg26: tensor<32xf32>, %arg27: tensor<32xf32>, %arg28: tensor<32x32xf32>, %arg29: tensor<32xf32>, %arg30: tensor<32xf32>, %arg31: tensor<32xf32>, %arg32: tensor<128x32xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128x1x3x3xf32>, %arg35: tensor<128xf32>, %arg36: tensor<32x128xf32>, %arg37: tensor<32xf32>, %arg38: tensor<32xf32>, %arg39: tensor<32xf32>, %arg40: tensor<32x32xf32>, %arg41: tensor<32xf32>, %arg42: tensor<32x32xf32>, %arg43: tensor<32xf32>, %arg44: tensor<32x32xf32>, %arg45: tensor<32xf32>, %arg46: tensor<32x32x8x8xf32>, %arg47: tensor<32xf32>, %arg48: tensor<32xf32>, %arg49: tensor<32xf32>, %arg50: tensor<32x32xf32>, %arg51: tensor<32xf32>, %arg52: tensor<32xf32>, %arg53: tensor<32xf32>, %arg54: tensor<128x32xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128x1x3x3xf32>, %arg57: tensor<128xf32>, %arg58: tensor<32x128xf32>, %arg59: tensor<32xf32>, %arg60: tensor<64xf32>, %arg61: tensor<64xf32>, %arg62: tensor<64x64xf32>, %arg63: tensor<64xf32>, %arg64: tensor<64x64xf32>, %arg65: tensor<64xf32>, %arg66: tensor<64x64xf32>, %arg67: tensor<64xf32>, %arg68: tensor<64x64x4x4xf32>, %arg69: tensor<64xf32>, %arg70: tensor<64xf32>, %arg71: tensor<64xf32>, %arg72: tensor<64x64xf32>, %arg73: tensor<64xf32>, %arg74: tensor<64xf32>, %arg75: tensor<64xf32>, %arg76: tensor<256x64xf32>, %arg77: tensor<256xf32>, %arg78: tensor<256x1x3x3xf32>, %arg79: tensor<256xf32>, %arg80: tensor<64x256xf32>, %arg81: tensor<64xf32>, %arg82: tensor<64xf32>, %arg83: tensor<64xf32>, %arg84: tensor<64x64xf32>, %arg85: tensor<64xf32>, %arg86: tensor<64x64xf32>, %arg87: tensor<64xf32>, %arg88: tensor<64x64xf32>, %arg89: tensor<64xf32>, %arg90: tensor<64x64x4x4xf32>, %arg91: tensor<64xf32>, %arg92: tensor<64xf32>, %arg93: tensor<64xf32>, %arg94: tensor<64x64xf32>, %arg95: tensor<64xf32>, %arg96: tensor<64xf32>, %arg97: tensor<64xf32>, %arg98: tensor<256x64xf32>, %arg99: tensor<256xf32>, %arg100: tensor<256x1x3x3xf32>, %arg101: tensor<256xf32>, %arg102: tensor<64x256xf32>, %arg103: tensor<64xf32>, %arg104: tensor<160xf32>, %arg105: tensor<160xf32>, %arg106: tensor<160x160xf32>, %arg107: tensor<160xf32>, %arg108: tensor<160x160xf32>, %arg109: tensor<160xf32>, %arg110: tensor<160x160xf32>, %arg111: tensor<160xf32>, %arg112: tensor<160x160x2x2xf32>, %arg113: tensor<160xf32>, %arg114: tensor<160xf32>, %arg115: tensor<160xf32>, %arg116: tensor<160x160xf32>, %arg117: tensor<160xf32>, %arg118: tensor<160xf32>, %arg119: tensor<160xf32>, %arg120: tensor<640x160xf32>, %arg121: tensor<640xf32>, %arg122: tensor<640x1x3x3xf32>, %arg123: tensor<640xf32>, %arg124: tensor<160x640xf32>, %arg125: tensor<160xf32>, %arg126: tensor<160xf32>, %arg127: tensor<160xf32>, %arg128: tensor<160x160xf32>, %arg129: tensor<160xf32>, %arg130: tensor<160x160xf32>, %arg131: tensor<160xf32>, %arg132: tensor<160x160xf32>, %arg133: tensor<160xf32>, %arg134: tensor<160x160x2x2xf32>, %arg135: tensor<160xf32>, %arg136: tensor<160xf32>, %arg137: tensor<160xf32>, %arg138: tensor<160x160xf32>, %arg139: tensor<160xf32>, %arg140: tensor<160xf32>, %arg141: tensor<160xf32>, %arg142: tensor<640x160xf32>, %arg143: tensor<640xf32>, %arg144: tensor<640x1x3x3xf32>, %arg145: tensor<640xf32>, %arg146: tensor<160x640xf32>, %arg147: tensor<160xf32>, %arg148: tensor<256xf32>, %arg149: tensor<256xf32>, %arg150: tensor<256x256xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256x256xf32>, %arg153: tensor<256xf32>, %arg154: tensor<256x256xf32>, %arg155: tensor<256xf32>, %arg156: tensor<256x256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<256xf32>, %arg160: tensor<1024x256xf32>, %arg161: tensor<1024xf32>, %arg162: tensor<1024x1x3x3xf32>, %arg163: tensor<1024xf32>, %arg164: tensor<256x1024xf32>, %arg165: tensor<256xf32>, %arg166: tensor<256xf32>, %arg167: tensor<256xf32>, %arg168: tensor<256x256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256x256xf32>, %arg171: tensor<256xf32>, %arg172: tensor<256x256xf32>, %arg173: tensor<256xf32>, %arg174: tensor<256x256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<256xf32>, %arg178: tensor<1024x256xf32>, %arg179: tensor<1024xf32>, %arg180: tensor<1024x1x3x3xf32>, %arg181: tensor<1024xf32>, %arg182: tensor<256x1024xf32>, %arg183: tensor<256xf32>, %arg184: tensor<32xf32>, %arg185: tensor<32xf32>, %arg186: tensor<64xf32>, %arg187: tensor<64xf32>, %arg188: tensor<160xf32>, %arg189: tensor<160xf32>, %arg190: tensor<256xf32>, %arg191: tensor<256xf32>, %arg192: tensor<256x32xf32>, %arg193: tensor<256xf32>, %arg194: tensor<256x64xf32>, %arg195: tensor<256xf32>, %arg196: tensor<256x160xf32>, %arg197: tensor<256xf32>, %arg198: tensor<256x256xf32>, %arg199: tensor<256xf32>, %arg200: tensor<256x1024x1x1xf32>, %arg201: tensor<256xf32>, %arg202: tensor<256xf32>, %arg203: tensor<150x256x1x1xf32>, %arg204: tensor<150xf32>, %arg205: tensor<256xf32>, %arg206: tensor<256xf32>, %arg207: tensor<64x3x512x512xf32>) -> tensor<64x150x128x128xf32> {
    %0 = "mhlo.fusion"(%arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<256xf32>
      %419 = stablehlo.rsqrt %418 : tensor<256xf32>
      mhlo.return %419 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<127> : tensor<i32>
      %417 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.iota dim = 0 : tensor<128xi32>
      %420 = stablehlo.convert %419 : (tensor<128xi32>) -> tensor<128xf32>
      %421 = stablehlo.maximum %418, %420 : tensor<128xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %423 = stablehlo.convert %422 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_4 = stablehlo.constant dense<1> : tensor<i32>
      %424 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %425 = stablehlo.add %423, %424 : tensor<128x1xi32>
      %426 = stablehlo.minimum %417, %425 : tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %427 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %428 = stablehlo.compare LT, %426, %427 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<128> : tensor<i32>
      %429 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %430 = stablehlo.add %426, %429 : tensor<128x1xi32>
      %431 = stablehlo.select %428, %430, %426 : tensor<128x1xi1>, tensor<128x1xi32>
      %432 = mhlo.bitcast %431 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %433 = stablehlo.broadcast_in_dim %432, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %434 = stablehlo.convert %421 : (tensor<128xf32>) -> tensor<128xi32>
      %435 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %437 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %438 = stablehlo.add %434, %437 : tensor<128xi32>
      %439 = stablehlo.select %436, %438, %434 : tensor<128xi1>, tensor<128xi32>
      %440 = stablehlo.broadcast_in_dim %439, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %441 = stablehlo.concatenate %433, %440, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %441 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.iota dim = 0 : tensor<128xi32>
      %419 = stablehlo.convert %418 : (tensor<128xi32>) -> tensor<128xf32>
      %420 = stablehlo.maximum %417, %419 : tensor<128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %422 = stablehlo.convert %421 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_3 = stablehlo.constant dense<0> : tensor<i32>
      %423 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %424 = stablehlo.compare LT, %422, %423 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_4 = stablehlo.constant dense<128> : tensor<i32>
      %425 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %426 = stablehlo.add %422, %425 : tensor<128x1xi32>
      %427 = stablehlo.select %424, %426, %422 : tensor<128x1xi1>, tensor<128x1xi32>
      %428 = mhlo.bitcast %427 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %430 = stablehlo.convert %420 : (tensor<128xf32>) -> tensor<128xi32>
      %431 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %432 = stablehlo.compare LT, %430, %431 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %433 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %434 = stablehlo.add %430, %433 : tensor<128xi32>
      %435 = stablehlo.select %432, %434, %430 : tensor<128xi1>, tensor<128xi32>
      %436 = stablehlo.broadcast_in_dim %435, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %437 = stablehlo.concatenate %429, %436, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %437 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x3x512x512xf32>):
      %417 = stablehlo.transpose %arg208, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,512,512,3]{2,1,3,0}"} : (tensor<64x3x512x512xf32>) -> tensor<64x512x512x3xf32>
      %418 = mhlo.copy %417 : tensor<64x512x512x3xf32>
      mhlo.return %418 : tensor<64x512x512x3xf32>
    }) {output_operand_aliasing = []} : (tensor<64x3x512x512xf32>) -> tensor<64x512x512x3xf32>
    %4 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x3x7x7xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,3,32]{1,0,2,3}"} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
      %418 = mhlo.copy %417 : tensor<7x7x3x32xf32>
      mhlo.return %418 : tensor<7x7x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x7x7xf32>) -> tensor<7x7x3x32xf32>
    %5 = stablehlo.convolution(%3, %4) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x512x512x3xf32>, tensor<7x7x3x32xf32>) -> tensor<64x128x128x32xf32>
    %6 = "mhlo.fusion"(%5, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x128x128x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x128x128x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x32x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %423 = mhlo.copy %422 : tensor<64x16384x32xf32>
      mhlo.return %423 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x128x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = "mhlo.fusion"(%6, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      mhlo.return %417 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %8 = "mhlo.fusion"(%7, %5, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384xf32>, %arg209: tensor<64x128x128x32xf32>, %arg210: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x128x128x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x32x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %423 = mhlo.copy %422 : tensor<64x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x16384xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x16384x32xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %429 = stablehlo.reduce(%428 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %430 = stablehlo.multiply %429, %424 : tensor<64x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %432 = stablehlo.add %430, %431 : tensor<64x16384xf32>
      %433 = stablehlo.rsqrt %432 : tensor<64x16384xf32>
      mhlo.return %433 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384xf32>, tensor<64x128x128x32xf32>, tensor<32xf32>) -> tensor<64x16384xf32>
    %9 = "mhlo.fusion"(%arg3, %arg2, %8, %7, %5, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384xf32>, %arg212: tensor<64x128x128x32xf32>, %arg213: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x128x128x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x32x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %423 = mhlo.copy %422 : tensor<64x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x16384xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x16384x32xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x16384x32xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x16384x32xf32>
      mhlo.return %433 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<64x128x128x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %10 = "mhlo.fusion"(%9, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      mhlo.return %417 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %11 = "mhlo.fusion"(%10, %arg3, %arg2, %8, %7, %5, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<64x16384xf32>, %arg212: tensor<64x16384xf32>, %arg213: tensor<64x128x128x32xf32>, %arg214: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %418 = stablehlo.add %arg213, %417 : tensor<64x128x128x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x32x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %423 = mhlo.copy %422 : tensor<64x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.multiply %arg212, %424 : tensor<64x16384xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x16384x32xf32>
      %430 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x16384x32xf32>
      %432 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x16384x32xf32>
      %434 = stablehlo.multiply %arg208, %424 : tensor<64x16384xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x16384x32xf32>
      %437 = stablehlo.multiply %436, %436 : tensor<64x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %438 = stablehlo.reduce(%437 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %439 = stablehlo.multiply %438, %424 : tensor<64x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x16384xf32>
      %442 = stablehlo.rsqrt %441 : tensor<64x16384xf32>
      mhlo.return %442 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<64x128x128x32xf32>, tensor<32xf32>) -> tensor<64x16384xf32>
    %12 = "mhlo.fusion"(%arg17, %arg16, %11, %10, %arg3, %arg2, %8, %7, %5, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<64x16384xf32>, %arg215: tensor<64x16384xf32>, %arg216: tensor<64x128x128x32xf32>, %arg217: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x128x128x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x32x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %423 = mhlo.copy %422 : tensor<64x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x16384xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x16384x32xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x16384x32xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x16384x32xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x16384xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x16384x32xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x16384x32xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x16384x32xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x16384x32xf32>
      %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %443 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<64x128x128x32xf32>, tensor<32xf32>) -> tensor<1048576x32xf32>
    %13 = stablehlo.dot_general %12, %arg18, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %14 = "mhlo.fusion"(%13, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<1048576x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      mhlo.return %419 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %15 = "mhlo.fusion"(%arg17, %arg16, %11, %10, %arg3, %arg2, %8, %7, %5, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384xf32>, %arg212: tensor<32xf32>, %arg213: tensor<32xf32>, %arg214: tensor<64x16384xf32>, %arg215: tensor<64x16384xf32>, %arg216: tensor<64x128x128x32xf32>, %arg217: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x128x128x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x32x128x128xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %423 = mhlo.copy %422 : tensor<64x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x16384xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x16384x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x16384x32xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x16384x32xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x16384x32xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x16384xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x16384x32xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x16384x32xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x16384x32xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x16384x32xf32>
      %443 = stablehlo.transpose %442, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,32,16384]{1,2,0}"} : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
      %444 = mhlo.copy %443 : tensor<64x32x16384xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
      %446 = stablehlo.transpose %445, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,32]{2,1,3,0}"} : (tensor<64x32x128x128xf32>) -> tensor<64x128x128x32xf32>
      %447 = mhlo.copy %446 : tensor<64x128x128x32xf32>
      mhlo.return %447 : tensor<64x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<64x128x128x32xf32>, tensor<32xf32>) -> tensor<64x128x128x32xf32>
    %16 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %418 = mhlo.copy %417 : tensor<8x8x32x32xf32>
      mhlo.return %418 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %17 = stablehlo.convolution(%15, %16) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<64x16x16x32xf32>
    %18 = "mhlo.fusion"(%17, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x32x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %423 = mhlo.copy %422 : tensor<64x256x32xf32>
      mhlo.return %423 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x32xf32>, tensor<32xf32>) -> tensor<64x256x32xf32>
    %19 = "mhlo.fusion"(%18, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
    %20 = "mhlo.fusion"(%19, %17, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x32xf32>, %arg210: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x32x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %423 = mhlo.copy %422 : tensor<64x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x32xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %429 = stablehlo.reduce(%428 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
      %430 = stablehlo.multiply %429, %424 : tensor<64x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %432 = stablehlo.add %430, %431 : tensor<64x256xf32>
      %433 = stablehlo.rsqrt %432 : tensor<64x256xf32>
      mhlo.return %433 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x32xf32>, tensor<32xf32>) -> tensor<64x256xf32>
    %21 = "mhlo.fusion"(%arg27, %arg26, %20, %19, %17, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x32xf32>, %arg213: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x32x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %423 = mhlo.copy %422 : tensor<64x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x32xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x32xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x32xf32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %434 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x32xf32>, tensor<32xf32>) -> tensor<16384x32xf32>
    %22 = stablehlo.dot_general %21, %arg20, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %23 = "mhlo.fusion"(%22, %arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,1,256,32]{3,2,0,1}"} : (tensor<16384x32xf32>) -> tensor<64x1x256x32xf32>
      %420 = mhlo.copy %419 : tensor<64x1x256x32xf32>
      %421 = stablehlo.transpose %420, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[64,1,32,256]{2,3,1,0}"} : (tensor<64x1x256x32xf32>) -> tensor<64x1x32x256xf32>
      %422 = mhlo.copy %421 : tensor<64x1x32x256xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x32x256xf32>) -> tensor<64x32x256xf32>
      mhlo.return %423 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>) -> tensor<64x32x256xf32>
    %24 = stablehlo.dot_general %14, %23, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
    %25 = "mhlo.fusion"(%24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
      %420 = mhlo.copy %419 : tensor<64x1x16384x256xf32>
      mhlo.return %420 : tensor<64x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %26 = "mhlo.fusion"(%cst, %25) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<f32>, %arg209: tensor<64x1x16384x256xf32>):
      %417 = "stablehlo.reduce_window"(%arg209, %arg208) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %419 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %419 : tensor<f32>
      }) : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384x8xf32>
      %418 = stablehlo.reduce(%417 init: %arg208) applies stablehlo.maximum across dimensions = [3] : (tensor<64x1x16384x8xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
      mhlo.return %418 : tensor<64x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<64x1x16384x256xf32>) -> tensor<64x1x16384xf32>
    %27 = "mhlo.fusion"(%26, %24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1x16384xf32>, %arg209: tensor<64x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
      %420 = mhlo.copy %419 : tensor<64x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1x16384xf32>
      %422 = stablehlo.maximum %421, %arg208 : tensor<64x1x16384xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<64x1x16384x256xf32>
      %426 = stablehlo.exponential %425 : tensor<64x1x16384x256xf32>
      mhlo.return %426 : tensor<64x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x16384xf32>, tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %28 = "mhlo.fusion"(%27, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1x16384x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
      mhlo.return %417 : tensor<64x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
    %29 = "mhlo.fusion"(%27, %28) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1x16384x256xf32>, %arg209: tensor<64x1x16384xf32>):
      %417 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
      %419 = stablehlo.divide %arg208, %418 : tensor<64x1x16384x256xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x16384x256xf32>) -> tensor<64x16384x256xf32>
      mhlo.return %420 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x16384x256xf32>, tensor<64x1x16384xf32>) -> tensor<64x16384x256xf32>
    %30 = stablehlo.dot_general %21, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %31 = "mhlo.fusion"(%30, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
      mhlo.return %419 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>) -> tensor<64x256x32xf32>
    %32 = stablehlo.dot_general %29, %31, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x256xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
    %33 = mhlo.bitcast %32 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %34 = stablehlo.dot_general %33, %arg28, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %35 = "mhlo.fusion"(%34, %arg3, %arg2, %8, %7, %5, %arg1, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<32xf32>, %arg211: tensor<64x16384xf32>, %arg212: tensor<64x16384xf32>, %arg213: tensor<64x128x128x32xf32>, %arg214: tensor<32xf32>, %arg215: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<1048576x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<32xf32>) -> tensor<64x128x128x32xf32>
      %421 = stablehlo.add %arg213, %420 : tensor<64x128x128x32xf32>
      %422 = stablehlo.transpose %421, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %423 = mhlo.copy %422 : tensor<64x32x128x128xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,32]{1,2,0}"} : (tensor<64x32x16384xf32>) -> tensor<64x16384x32xf32>
      %426 = mhlo.copy %425 : tensor<64x16384x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %428 = stablehlo.multiply %arg212, %427 : tensor<64x16384xf32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %430 = stablehlo.subtract %426, %429 : tensor<64x16384x32xf32>
      %431 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<64x16384x32xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<64x16384x32xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %436 = stablehlo.add %434, %435 : tensor<64x16384x32xf32>
      %437 = stablehlo.add %419, %436 : tensor<64x16384x32xf32>
      mhlo.return %437 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384xf32>, tensor<64x128x128x32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %36 = "mhlo.fusion"(%35, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      mhlo.return %417 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %37 = "mhlo.fusion"(%35, %36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %422 = stablehlo.reduce(%421 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %423 = stablehlo.multiply %422, %417 : tensor<64x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.add %423, %424 : tensor<64x16384xf32>
      %426 = stablehlo.rsqrt %425 : tensor<64x16384xf32>
      mhlo.return %426 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x16384xf32>
    %38 = "mhlo.fusion"(%arg31, %arg30, %37, %35, %36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384x32xf32>, %arg212: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x16384x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %427 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<1048576x32xf32>
    %39 = stablehlo.dot_general %38, %arg32, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
    %40 = "mhlo.fusion"(%39, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x128xf32>, %arg209: tensor<128xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<1048576x128xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,128,16384]{1,2,0}"} : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
      %421 = mhlo.copy %420 : tensor<64x128x16384xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x16384xf32>) -> tensor<64x128x128x128xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{2,1,3,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
      %424 = mhlo.copy %423 : tensor<64x128x128x128xf32>
      mhlo.return %424 : tensor<64x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x128xf32>, tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %41 = "mhlo.fusion"(%arg34) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x128xf32>
      mhlo.return %418 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %42 = stablehlo.convolution(%40, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<64x128x128x128xf32>
    %43 = "mhlo.fusion"(%42, %arg35) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x128x128x128xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x128x128x128xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
      %422 = mhlo.copy %421 : tensor<64x128x128x128xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
      %425 = mhlo.copy %424 : tensor<64x16384x128xf32>
      %426 = stablehlo.negate %419 : tensor<64x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x128x128x128xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
      %430 = mhlo.copy %429 : tensor<64x128x128x128xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
      %433 = mhlo.copy %432 : tensor<64x16384x128xf32>
      %434 = stablehlo.abs %433 : tensor<64x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x16384x128xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x16384x128xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x16384x128xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x16384x128xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x16384x128xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x16384x128xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x16384x128xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x16384x128xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %462 = stablehlo.negate %438 : tensor<64x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %465 = stablehlo.exponential %462 : tensor<64x16384x128xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x16384x128xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x16384x128xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x16384x128xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x16384x128xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x16384x128xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x16384x128xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x16384x128xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x16384x128xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x16384x128xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x16384x128xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x16384x128xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x16384x128xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x16384x128xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x16384x128xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x16384x128xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x16384x128xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x16384x128xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x16384x128xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x16384x128xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x16384x128xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
      mhlo.return %524 : tensor<1048576x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x128x128xf32>, tensor<128xf32>) -> tensor<1048576x128xf32>
    %44 = stablehlo.dot_general %43, %arg36, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x128xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
    %45 = "mhlo.fusion"(%35, %44, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<1048576x32xf32>, %arg210: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<1048576x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x16384x32xf32>
      mhlo.return %420 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %46 = "mhlo.fusion"(%45, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      mhlo.return %417 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %47 = "mhlo.fusion"(%45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %422 = stablehlo.reduce(%421 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %423 = stablehlo.multiply %422, %417 : tensor<64x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.add %423, %424 : tensor<64x16384xf32>
      %426 = stablehlo.rsqrt %425 : tensor<64x16384xf32>
      mhlo.return %426 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x16384xf32>
    %48 = "mhlo.fusion"(%arg39, %arg38, %47, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384x32xf32>, %arg212: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x16384x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %427 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<1048576x32xf32>
    %49 = stablehlo.dot_general %48, %arg40, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %50 = "mhlo.fusion"(%49, %arg41) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<1048576x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      mhlo.return %419 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %51 = "mhlo.fusion"(%arg39, %arg38, %47, %45, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384x32xf32>, %arg212: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x16384x32xf32>
      %427 = stablehlo.transpose %426, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,32,16384]{1,2,0}"} : (tensor<64x16384x32xf32>) -> tensor<64x32x16384xf32>
      %428 = mhlo.copy %427 : tensor<64x32x16384xf32>
      %429 = mhlo.bitcast %428 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x32x16384xf32>) -> tensor<64x32x128x128xf32>
      %430 = stablehlo.transpose %429, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,32]{2,1,3,0}"} : (tensor<64x32x128x128xf32>) -> tensor<64x128x128x32xf32>
      %431 = mhlo.copy %430 : tensor<64x128x128x32xf32>
      mhlo.return %431 : tensor<64x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x128x128x32xf32>
    %52 = "mhlo.fusion"(%arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32x32x8x8xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[8,8,32,32]{1,0,2,3}"} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
      %418 = mhlo.copy %417 : tensor<8x8x32x32xf32>
      mhlo.return %418 : tensor<8x8x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x32x8x8xf32>) -> tensor<8x8x32x32xf32>
    %53 = stablehlo.convolution(%51, %52) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [8, 8], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x128x128x32xf32>, tensor<8x8x32x32xf32>) -> tensor<64x16x16x32xf32>
    %54 = "mhlo.fusion"(%53, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x32x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %423 = mhlo.copy %422 : tensor<64x256x32xf32>
      mhlo.return %423 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x32xf32>, tensor<32xf32>) -> tensor<64x256x32xf32>
    %55 = "mhlo.fusion"(%54, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
    %56 = "mhlo.fusion"(%55, %53, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x32xf32>, %arg210: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x32x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %423 = mhlo.copy %422 : tensor<64x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x32xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %429 = stablehlo.reduce(%428 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x256x32xf32>, tensor<f32>) -> tensor<64x256xf32>
      %430 = stablehlo.multiply %429, %424 : tensor<64x256xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %431 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %432 = stablehlo.add %430, %431 : tensor<64x256xf32>
      %433 = stablehlo.rsqrt %432 : tensor<64x256xf32>
      mhlo.return %433 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x32xf32>, tensor<32xf32>) -> tensor<64x256xf32>
    %57 = "mhlo.fusion"(%arg49, %arg48, %56, %55, %53, %arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x32xf32>, %arg213: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<32xf32>) -> tensor<64x16x16x32xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x32xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,16,16]{1,3,2,0}"} : (tensor<64x16x16x32xf32>) -> tensor<64x32x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x32x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x16x16xf32>) -> tensor<64x32x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,32]{1,2,0}"} : (tensor<64x32x256xf32>) -> tensor<64x256x32xf32>
      %423 = mhlo.copy %422 : tensor<64x256x32xf32>
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x32xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x32xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x32xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x32xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x256x32xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x32xf32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x32xf32>) -> tensor<16384x32xf32>
      mhlo.return %434 : tensor<16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x32xf32>, tensor<32xf32>) -> tensor<16384x32xf32>
    %58 = stablehlo.dot_general %57, %arg42, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %59 = "mhlo.fusion"(%58, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[64,1,256,32]{3,2,0,1}"} : (tensor<16384x32xf32>) -> tensor<64x1x256x32xf32>
      %420 = mhlo.copy %419 : tensor<64x1x256x32xf32>
      %421 = stablehlo.transpose %420, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[64,1,32,256]{2,3,1,0}"} : (tensor<64x1x256x32xf32>) -> tensor<64x1x32x256xf32>
      %422 = mhlo.copy %421 : tensor<64x1x32x256xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x32x256xf32>) -> tensor<64x32x256xf32>
      mhlo.return %423 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>) -> tensor<64x32x256xf32>
    %60 = stablehlo.dot_general %50, %59, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x32xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
      %420 = mhlo.copy %419 : tensor<64x1x16384x256xf32>
      mhlo.return %420 : tensor<64x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %62 = "mhlo.fusion"(%cst, %61) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<f32>, %arg209: tensor<64x1x16384x256xf32>):
      %417 = "stablehlo.reduce_window"(%arg209, %arg208) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %419 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %419 : tensor<f32>
      }) : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384x8xf32>
      %418 = stablehlo.reduce(%417 init: %arg208) applies stablehlo.maximum across dimensions = [3] : (tensor<64x1x16384x8xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
      mhlo.return %418 : tensor<64x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>, tensor<64x1x16384x256xf32>) -> tensor<64x1x16384xf32>
    %63 = "mhlo.fusion"(%62, %60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1x16384xf32>, %arg209: tensor<64x16384x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>, xla_shape = "f32[64,1,16384,256]{3,2,0,1}"} : (tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
      %420 = mhlo.copy %419 : tensor<64x1x16384x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1x16384xf32>
      %422 = stablehlo.maximum %421, %arg208 : tensor<64x1x16384xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
      %424 = stablehlo.broadcast_in_dim %423, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
      %425 = stablehlo.subtract %420, %424 : tensor<64x1x16384x256xf32>
      %426 = stablehlo.exponential %425 : tensor<64x1x16384x256xf32>
      mhlo.return %426 : tensor<64x1x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x16384xf32>, tensor<64x16384x256xf32>) -> tensor<64x1x16384x256xf32>
    %64 = "mhlo.fusion"(%63, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1x16384x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
      mhlo.return %417 : tensor<64x1x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x16384x256xf32>, tensor<f32>) -> tensor<64x1x16384xf32>
    %65 = "mhlo.fusion"(%63, %64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1x16384x256xf32>, %arg209: tensor<64x1x16384xf32>):
      %417 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1x16384xf32>) -> tensor<64x16384xf32>
      %418 = stablehlo.broadcast_in_dim %417, dims = [0, 2] : (tensor<64x16384xf32>) -> tensor<64x1x16384x256xf32>
      %419 = stablehlo.divide %arg208, %418 : tensor<64x1x16384x256xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x16384x256xf32>) -> tensor<64x16384x256xf32>
      mhlo.return %420 : tensor<64x16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x16384x256xf32>, tensor<64x1x16384xf32>) -> tensor<64x16384x256xf32>
    %66 = stablehlo.dot_general %57, %arg44, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x32xf32>, tensor<32x32xf32>) -> tensor<16384x32xf32>
    %67 = "mhlo.fusion"(%66, %arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x32xf32>, %arg209: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<32xf32>) -> tensor<16384x32xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x32xf32>) -> tensor<64x256x32xf32>
      mhlo.return %419 : tensor<64x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x32xf32>, tensor<32xf32>) -> tensor<64x256x32xf32>
    %68 = stablehlo.dot_general %65, %67, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x16384x256xf32>, tensor<64x256x32xf32>) -> tensor<64x16384x32xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
    %70 = stablehlo.dot_general %69, %arg50, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<32x32xf32>) -> tensor<1048576x32xf32>
    %71 = "mhlo.fusion"(%45, %70, %arg51) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<1048576x32xf32>, %arg210: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<1048576x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x16384x32xf32>
      mhlo.return %420 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %72 = "mhlo.fusion"(%71, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      mhlo.return %417 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %73 = "mhlo.fusion"(%71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %422 = stablehlo.reduce(%421 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %423 = stablehlo.multiply %422, %417 : tensor<64x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.add %423, %424 : tensor<64x16384xf32>
      %426 = stablehlo.rsqrt %425 : tensor<64x16384xf32>
      mhlo.return %426 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x16384xf32>
    %74 = "mhlo.fusion"(%arg53, %arg52, %73, %71, %72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384x32xf32>, %arg212: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x16384x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<1048576x32xf32>
      mhlo.return %427 : tensor<1048576x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<1048576x32xf32>
    %75 = stablehlo.dot_general %74, %arg54, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x32xf32>, tensor<128x32xf32>) -> tensor<1048576x128xf32>
    %76 = "mhlo.fusion"(%75, %arg55) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1048576x128xf32>, %arg209: tensor<128xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<128xf32>) -> tensor<1048576x128xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<1048576x128xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x128xf32>) -> tensor<64x16384x128xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,128,16384]{1,2,0}"} : (tensor<64x16384x128xf32>) -> tensor<64x128x16384xf32>
      %421 = mhlo.copy %420 : tensor<64x128x16384xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x128x16384xf32>) -> tensor<64x128x128x128xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{2,1,3,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
      %424 = mhlo.copy %423 : tensor<64x128x128x128xf32>
      mhlo.return %424 : tensor<64x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1048576x128xf32>, tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %77 = "mhlo.fusion"(%arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x128xf32>
      mhlo.return %418 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %78 = stablehlo.convolution(%76, %77) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<64x128x128x128xf32>
    %79 = "mhlo.fusion"(%78, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x128x128x128xf32>, %arg209: tensor<128xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x128x128x128xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x128x128x128xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
      %422 = mhlo.copy %421 : tensor<64x128x128x128xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
      %425 = mhlo.copy %424 : tensor<64x16384x128xf32>
      %426 = stablehlo.negate %419 : tensor<64x128x128x128xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x128x128x128xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,128]{1,3,2,0}"} : (tensor<64x128x128x128xf32>) -> tensor<64x128x128x128xf32>
      %430 = mhlo.copy %429 : tensor<64x128x128x128xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x128xf32>) -> tensor<64x128x16384xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,16384,128]{1,2,0}"} : (tensor<64x128x16384xf32>) -> tensor<64x16384x128xf32>
      %433 = mhlo.copy %432 : tensor<64x16384x128xf32>
      %434 = stablehlo.abs %433 : tensor<64x16384x128xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x16384x128xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x16384x128xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x16384x128xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x16384x128xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x16384x128xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x16384x128xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x16384x128xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x16384x128xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x16384x128xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x16384x128xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x16384x128xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x16384x128xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x16384x128xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x16384x128xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x16384x128xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %462 = stablehlo.negate %438 : tensor<64x16384x128xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %465 = stablehlo.exponential %462 : tensor<64x16384x128xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x16384x128xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x16384x128xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x16384x128xf32>, tensor<64x16384x128xf32>) -> tensor<64x16384x128xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x16384x128xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x16384x128xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x16384x128xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x16384x128xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x16384x128xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x16384x128xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x16384x128xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x16384x128xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x16384x128xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x16384x128xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x16384x128xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x16384x128xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x16384x128xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x16384x128xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x16384x128xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x16384x128xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x16384x128xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x16384x128xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x16384x128xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x16384x128xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x16384x128xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x16384x128xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x16384x128xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x16384x128xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x16384x128xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x16384x128xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x16384x128xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x16384x128xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x16384x128xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x16384x128xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x16384x128xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x16384x128xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x16384x128xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x16384x128xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x16384x128xi1>, tensor<64x16384x128xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x16384x128xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x128xf32>) -> tensor<1048576x128xf32>
      mhlo.return %524 : tensor<1048576x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x128x128xf32>, tensor<128xf32>) -> tensor<1048576x128xf32>
    %80 = stablehlo.dot_general %79, %arg58, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1048576x128xf32>, tensor<32x128xf32>) -> tensor<1048576x32xf32>
    %81 = "mhlo.fusion"(%71, %80, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<1048576x32xf32>, %arg210: tensor<32xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<32xf32>) -> tensor<1048576x32xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<1048576x32xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x32xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x16384x32xf32>
      mhlo.return %420 : tensor<64x16384x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<1048576x32xf32>, tensor<32xf32>) -> tensor<64x16384x32xf32>
    %82 = "mhlo.fusion"(%81, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      mhlo.return %417 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
    %83 = "mhlo.fusion"(%81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16384x32xf32>, %arg209: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x16384x32xf32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %422 = stablehlo.reduce(%421 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<64x16384x32xf32>, tensor<f32>) -> tensor<64x16384xf32>
      %423 = stablehlo.multiply %422, %417 : tensor<64x16384xf32>
      %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %425 = stablehlo.add %423, %424 : tensor<64x16384xf32>
      %426 = stablehlo.rsqrt %425 : tensor<64x16384xf32>
      mhlo.return %426 : tensor<64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x16384xf32>
    %84 = "mhlo.fusion"(%arg185, %arg184, %83, %81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384x32xf32>, %arg212: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x16384x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<64x128x128x32xf32>
      %428 = stablehlo.transpose %427, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,128,128]{1,3,2,0}"} : (tensor<64x128x128x32xf32>) -> tensor<64x32x128x128xf32>
      %429 = mhlo.copy %428 : tensor<64x32x128x128xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x32x128x128xf32>) -> tensor<64x32x16384xf32>
      mhlo.return %430 : tensor<64x32x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x32x16384xf32>
    %85 = "mhlo.fusion"(%arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x32xf32>):
      %417 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[32,256]{0,1}"} : (tensor<256x32xf32>) -> tensor<32x256xf32>
      %418 = mhlo.copy %417 : tensor<32x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [1, 2] : (tensor<32x256xf32>) -> tensor<64x32x256xf32>
      mhlo.return %419 : tensor<64x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x32xf32>) -> tensor<64x32x256xf32>
    %86 = stablehlo.dot_general %84, %85, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x32x16384xf32>, tensor<64x32x256xf32>) -> tensor<64x16384x256xf32>
    %87 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<63> : tensor<i32>
      %417 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.iota dim = 0 : tensor<128xi32>
      %420 = stablehlo.convert %419 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %422 = stablehlo.add %420, %421 : tensor<128xf32>
      %423 = stablehlo.multiply %422, %421 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %425 = stablehlo.add %423, %424 : tensor<128xf32>
      %426 = stablehlo.maximum %418, %425 : tensor<128xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %428 = stablehlo.convert %427 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<1> : tensor<i32>
      %429 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %430 = stablehlo.add %428, %429 : tensor<128x1xi32>
      %431 = stablehlo.minimum %417, %430 : tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<0> : tensor<i32>
      %432 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %433 = stablehlo.compare LT, %431, %432 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<64> : tensor<i32>
      %434 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %435 = stablehlo.add %431, %434 : tensor<128x1xi32>
      %436 = stablehlo.select %433, %435, %431 : tensor<128x1xi1>, tensor<128x1xi32>
      %437 = mhlo.bitcast %436 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %438 = stablehlo.broadcast_in_dim %437, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %439 = stablehlo.convert %426 : (tensor<128xf32>) -> tensor<128xi32>
      %440 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %441 = stablehlo.compare LT, %439, %440 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %442 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %443 = stablehlo.add %439, %442 : tensor<128xi32>
      %444 = stablehlo.select %441, %443, %439 : tensor<128xi1>, tensor<128xi32>
      %445 = stablehlo.broadcast_in_dim %444, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %446 = stablehlo.concatenate %438, %445, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %446 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %88 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.iota dim = 0 : tensor<128xi32>
      %419 = stablehlo.convert %418 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %422 = stablehlo.multiply %421, %420 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %424 = stablehlo.add %422, %423 : tensor<128xf32>
      %425 = stablehlo.maximum %417, %424 : tensor<128xf32>
      %426 = mhlo.bitcast %425 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %427 = stablehlo.convert %426 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_5 = stablehlo.constant dense<0> : tensor<i32>
      %428 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %429 = stablehlo.compare LT, %427, %428 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_6 = stablehlo.constant dense<64> : tensor<i32>
      %430 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %431 = stablehlo.add %427, %430 : tensor<128x1xi32>
      %432 = stablehlo.select %429, %431, %427 : tensor<128x1xi1>, tensor<128x1xi32>
      %433 = mhlo.bitcast %432 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %434 = stablehlo.broadcast_in_dim %433, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %435 = stablehlo.convert %425 : (tensor<128xf32>) -> tensor<128xi32>
      %436 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %438 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %439 = stablehlo.add %435, %438 : tensor<128xi32>
      %440 = stablehlo.select %437, %439, %435 : tensor<128xi1>, tensor<128xi32>
      %441 = stablehlo.broadcast_in_dim %440, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %442 = stablehlo.concatenate %434, %441, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %442 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %89 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %417 = stablehlo.iota dim = 0 : tensor<128xi32>
      %418 = stablehlo.convert %417 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %420 = stablehlo.add %418, %419 : tensor<128xf32>
      mhlo.return %420 : tensor<128xf32>
    }) {output_operand_aliasing = []} : () -> tensor<128xf32>
    %90 = "mhlo.fusion"(%arg185, %arg184, %83, %81, %82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<32xf32>, %arg209: tensor<32xf32>, %arg210: tensor<64x16384xf32>, %arg211: tensor<64x16384x32xf32>, %arg212: tensor<64x16384xf32>):
      %cst_2 = stablehlo.constant dense<3.125000e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16384xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x16384xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x16384x32xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x16384xf32>) -> tensor<64x16384x32xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x16384x32xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x16384x32xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<32xf32>) -> tensor<64x16384x32xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x16384x32xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x16384x32xf32>) -> tensor<64x128x128x32xf32>
      mhlo.return %427 : tensor<64x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<64x16384xf32>, tensor<64x16384x32xf32>, tensor<64x16384xf32>) -> tensor<64x128x128x32xf32>
    %91 = "mhlo.fusion"(%arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,32,64]{1,0,2,3}"} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
      %418 = mhlo.copy %417 : tensor<3x3x32x64xf32>
      mhlo.return %418 : tensor<3x3x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x3x3xf32>) -> tensor<3x3x32x64xf32>
    %92 = stablehlo.convolution(%90, %91) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x128x128x32xf32>, tensor<3x3x32x64xf32>) -> tensor<64x64x64x64xf32>
    %93 = "mhlo.fusion"(%92, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x64x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x64x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %420 = mhlo.copy %419 : tensor<64x64x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %423 = mhlo.copy %422 : tensor<64x4096x64xf32>
      mhlo.return %423 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x64x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %94 = "mhlo.fusion"(%93, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %95 = "mhlo.fusion"(%94, %92, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>, %arg209: tensor<64x64x64x64xf32>, %arg210: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x64x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %420 = mhlo.copy %419 : tensor<64x64x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %423 = mhlo.copy %422 : tensor<64x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x4096xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x4096x64xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x4096x64xf32>
      mhlo.return %428 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>, tensor<64x64x64x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %96 = "mhlo.fusion"(%95, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
      mhlo.return %417 : tensor<64x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    %97 = "mhlo.fusion"(%96, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %98 = "mhlo.fusion"(%97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x4096xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x4096xf32>
      mhlo.return %421 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>) -> tensor<64x4096xf32>
    %99 = "mhlo.fusion"(%arg7, %arg6, %98, %94, %92, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096xf32>, %arg212: tensor<64x64x64x64xf32>, %arg213: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x64x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %420 = mhlo.copy %419 : tensor<64x64x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %423 = mhlo.copy %422 : tensor<64x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x4096xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x4096x64xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x4096x64xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x4096x64xf32>
      mhlo.return %433 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64x64x64x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %100 = "mhlo.fusion"(%99, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %101 = "mhlo.fusion"(%100, %arg7, %arg6, %98, %94, %92, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x4096xf32>, %arg212: tensor<64x4096xf32>, %arg213: tensor<64x64x64x64xf32>, %arg214: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %418 = stablehlo.add %arg213, %417 : tensor<64x64x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %420 = mhlo.copy %419 : tensor<64x64x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %423 = mhlo.copy %422 : tensor<64x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %425 = stablehlo.multiply %arg212, %424 : tensor<64x4096xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x4096x64xf32>
      %430 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x4096x64xf32>
      %432 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x4096x64xf32>
      %434 = stablehlo.multiply %arg208, %424 : tensor<64x4096xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x4096x64xf32>
      %437 = stablehlo.multiply %436, %436 : tensor<64x4096x64xf32>
      mhlo.return %437 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64x64x64x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %102 = "mhlo.fusion"(%101, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
      mhlo.return %417 : tensor<64x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    %103 = "mhlo.fusion"(%102, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x4096xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x4096xf32>
      mhlo.return %421 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>) -> tensor<64x4096xf32>
    %105 = "mhlo.fusion"(%arg61, %arg60, %104, %100, %arg7, %arg6, %98, %94, %92, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<64x4096xf32>, %arg215: tensor<64x4096xf32>, %arg216: tensor<64x64x64x64xf32>, %arg217: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x64x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %420 = mhlo.copy %419 : tensor<64x64x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %423 = mhlo.copy %422 : tensor<64x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x4096xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x4096x64xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x4096x64xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x4096x64xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x4096xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x4096x64xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x4096x64xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x4096x64xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x4096x64xf32>
      %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %443 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64x64x64x64xf32>, tensor<64xf32>) -> tensor<262144x64xf32>
    %106 = stablehlo.dot_general %105, %arg62, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %107 = "mhlo.fusion"(%106, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<262144x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,4096,32]{3,1,2,0}"} : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
      %421 = mhlo.copy %420 : tensor<64x2x4096x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x32xf32>) -> tensor<128x4096x32xf32>
      mhlo.return %422 : tensor<128x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>) -> tensor<128x4096x32xf32>
    %108 = "mhlo.fusion"(%arg61, %arg60, %104, %100, %arg7, %arg6, %98, %94, %92, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096xf32>, %arg212: tensor<64xf32>, %arg213: tensor<64xf32>, %arg214: tensor<64x4096xf32>, %arg215: tensor<64x4096xf32>, %arg216: tensor<64x64x64x64xf32>, %arg217: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x64x64x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %420 = mhlo.copy %419 : tensor<64x64x64x64xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %423 = mhlo.copy %422 : tensor<64x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x4096xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x4096x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x4096x64xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x4096x64xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x4096x64xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x4096xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x4096x64xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x4096x64xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x4096x64xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x4096x64xf32>
      %443 = stablehlo.transpose %442, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,64,4096]{1,2,0}"} : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
      %444 = mhlo.copy %443 : tensor<64x64x4096xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
      %446 = stablehlo.transpose %445, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{2,1,3,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %447 = mhlo.copy %446 : tensor<64x64x64x64xf32>
      mhlo.return %447 : tensor<64x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64x64x64x64xf32>, tensor<64xf32>) -> tensor<64x64x64x64xf32>
    %109 = "mhlo.fusion"(%arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %418 = mhlo.copy %417 : tensor<4x4x64x64xf32>
      mhlo.return %418 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %110 = stablehlo.convolution(%108, %109) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<64x16x16x64xf32>
    %111 = "mhlo.fusion"(%110, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x64x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %423 = mhlo.copy %422 : tensor<64x256x64xf32>
      mhlo.return %423 : tensor<64x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x64xf32>, tensor<64xf32>) -> tensor<64x256x64xf32>
    %112 = "mhlo.fusion"(%111, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256xf32>
    %113 = "mhlo.fusion"(%112, %110, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x64xf32>, %arg210: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x64x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %423 = mhlo.copy %422 : tensor<64x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x64xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x64xf32>
      mhlo.return %428 : tensor<64x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x64xf32>, tensor<64xf32>) -> tensor<64x256x64xf32>
    %114 = "mhlo.fusion"(%113, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256x2xf32>
      mhlo.return %417 : tensor<64x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256x2xf32>
    %115 = "mhlo.fusion"(%114, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x2xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x2xf32>, tensor<f32>) -> tensor<64x256xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %117 = "mhlo.fusion"(%arg71, %arg70, %116, %112, %110, %arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x64xf32>, %arg213: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x64x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %423 = mhlo.copy %422 : tensor<64x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x64xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x64xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x64xf32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
      mhlo.return %434 : tensor<16384x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x64xf32>, tensor<64xf32>) -> tensor<16384x64xf32>
    %118 = stablehlo.dot_general %117, %arg64, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x64xf32>) -> tensor<16384x64xf32>
    %119 = "mhlo.fusion"(%118, %arg65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,32,256]{2,1,3,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
      %421 = mhlo.copy %420 : tensor<64x2x32x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x32x256xf32>) -> tensor<128x32x256xf32>
      mhlo.return %422 : tensor<128x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>, tensor<64xf32>) -> tensor<128x32x256xf32>
    %120 = stablehlo.dot_general %107, %119, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
    %121 = "mhlo.fusion"(%120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<128x4096x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
      mhlo.return %419 : tensor<64x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %122 = "mhlo.fusion"(%121, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x2x4096x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096x8xf32>
      mhlo.return %417 : tensor<64x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096x8xf32>
    %123 = "mhlo.fusion"(%122, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x2x4096x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x4096x8xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
      mhlo.return %417 : tensor<64x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x8xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    %124 = "mhlo.fusion"(%123, %120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x2x4096xf32>, %arg209: tensor<128x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<128x4096x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x2x4096xf32>
      %421 = stablehlo.maximum %420, %arg208 : tensor<64x2x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<64x2x4096x256xf32>
      %424 = stablehlo.exponential %423 : tensor<64x2x4096x256xf32>
      mhlo.return %424 : tensor<64x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096xf32>, tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %125 = "mhlo.fusion"(%124, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x2x4096x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
      mhlo.return %417 : tensor<64x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    %126 = "mhlo.fusion"(%124, %125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x2x4096x256xf32>, %arg209: tensor<64x2x4096xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %418 = stablehlo.divide %arg208, %417 : tensor<64x2x4096x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x256xf32>) -> tensor<128x4096x256xf32>
      mhlo.return %419 : tensor<128x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x256xf32>, tensor<64x2x4096xf32>) -> tensor<128x4096x256xf32>
    %127 = stablehlo.dot_general %117, %arg66, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x64xf32>) -> tensor<16384x64xf32>
    %128 = "mhlo.fusion"(%127, %arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,256,32]{3,1,2,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x2x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x256x32xf32>) -> tensor<128x256x32xf32>
      mhlo.return %422 : tensor<128x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>, tensor<64xf32>) -> tensor<128x256x32xf32>
    %129 = stablehlo.dot_general %126, %128, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
    %130 = "mhlo.fusion"(%129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x4096x32xf32>):
      %417 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,4096,2,32]{3,1,2,0}"} : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
      %419 = mhlo.copy %418 : tensor<64x4096x2x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x4096x2x32xf32>) -> tensor<262144x64xf32>
      mhlo.return %420 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x32xf32>) -> tensor<262144x64xf32>
    %131 = stablehlo.dot_general %130, %arg72, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %132 = "mhlo.fusion"(%131, %arg7, %arg6, %98, %94, %92, %arg5, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64xf32>, %arg211: tensor<64x4096xf32>, %arg212: tensor<64x4096xf32>, %arg213: tensor<64x64x64x64xf32>, %arg214: tensor<64xf32>, %arg215: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<262144x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<64xf32>) -> tensor<64x64x64x64xf32>
      %421 = stablehlo.add %arg213, %420 : tensor<64x64x64x64xf32>
      %422 = stablehlo.transpose %421, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %423 = mhlo.copy %422 : tensor<64x64x64x64xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,64]{1,2,0}"} : (tensor<64x64x4096xf32>) -> tensor<64x4096x64xf32>
      %426 = mhlo.copy %425 : tensor<64x4096x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %428 = stablehlo.multiply %arg212, %427 : tensor<64x4096xf32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %430 = stablehlo.subtract %426, %429 : tensor<64x4096x64xf32>
      %431 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<64x4096x64xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<64x4096x64xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %436 = stablehlo.add %434, %435 : tensor<64x4096x64xf32>
      %437 = stablehlo.add %419, %436 : tensor<64x4096x64xf32>
      mhlo.return %437 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096xf32>, tensor<64x64x64x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %133 = "mhlo.fusion"(%132, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %134 = "mhlo.fusion"(%132, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x4096x64xf32>
      mhlo.return %421 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %135 = "mhlo.fusion"(%134, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
      mhlo.return %417 : tensor<64x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    %136 = "mhlo.fusion"(%135, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %137 = "mhlo.fusion"(%136) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x4096xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x4096xf32>
      mhlo.return %421 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>) -> tensor<64x4096xf32>
    %138 = "mhlo.fusion"(%arg75, %arg74, %137, %132, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096x64xf32>, %arg212: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x4096x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %427 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<262144x64xf32>
    %139 = stablehlo.dot_general %138, %arg76, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
    %140 = "mhlo.fusion"(%139, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<262144x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,4096]{1,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
      %421 = mhlo.copy %420 : tensor<64x256x4096xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,256]{2,1,3,0}"} : (tensor<64x256x64x64xf32>) -> tensor<64x64x64x256xf32>
      %424 = mhlo.copy %423 : tensor<64x64x64x256xf32>
      mhlo.return %424 : tensor<64x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x256xf32>, tensor<256xf32>) -> tensor<64x64x64x256xf32>
    %141 = "mhlo.fusion"(%arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x256xf32>
      mhlo.return %418 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %142 = stablehlo.convolution(%140, %141) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<64x64x64x256xf32>
    %143 = "mhlo.fusion"(%142, %arg79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<64x64x64x256xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x64x64x256xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x64x64x256xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
      %422 = mhlo.copy %421 : tensor<64x256x64x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
      %425 = mhlo.copy %424 : tensor<64x4096x256xf32>
      %426 = stablehlo.negate %419 : tensor<64x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x64x64x256xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
      %430 = mhlo.copy %429 : tensor<64x256x64x64xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
      %433 = mhlo.copy %432 : tensor<64x4096x256xf32>
      %434 = stablehlo.abs %433 : tensor<64x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x4096x256xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x4096x256xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x4096x256xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x4096x256xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x4096x256xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x4096x256xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x4096x256xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x4096x256xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %462 = stablehlo.negate %438 : tensor<64x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %465 = stablehlo.exponential %462 : tensor<64x4096x256xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x4096x256xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x4096x256xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x4096x256xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x4096x256xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x4096x256xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x4096x256xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x4096x256xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x4096x256xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x4096x256xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x4096x256xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x4096x256xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x4096x256xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x4096x256xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x4096x256xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x4096x256xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x4096x256xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x4096x256xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x4096x256xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x4096x256xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x4096x256xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
      mhlo.return %524 : tensor<262144x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x64x256xf32>, tensor<256xf32>) -> tensor<262144x256xf32>
    %144 = stablehlo.dot_general %143, %arg80, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
    %145 = "mhlo.fusion"(%132, %144, %arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<262144x64xf32>, %arg210: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<262144x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x4096x64xf32>
      mhlo.return %420 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %146 = "mhlo.fusion"(%145, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %147 = "mhlo.fusion"(%145, %146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x4096x64xf32>
      mhlo.return %421 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %148 = "mhlo.fusion"(%147, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
      mhlo.return %417 : tensor<64x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    %149 = "mhlo.fusion"(%148, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %150 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x4096xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x4096xf32>
      mhlo.return %421 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>) -> tensor<64x4096xf32>
    %151 = "mhlo.fusion"(%arg83, %arg82, %150, %145, %146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096x64xf32>, %arg212: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x4096x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %427 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<262144x64xf32>
    %152 = stablehlo.dot_general %151, %arg84, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %153 = "mhlo.fusion"(%152, %arg85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<262144x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x2x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,4096,32]{3,1,2,0}"} : (tensor<64x4096x2x32xf32>) -> tensor<64x2x4096x32xf32>
      %421 = mhlo.copy %420 : tensor<64x2x4096x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x32xf32>) -> tensor<128x4096x32xf32>
      mhlo.return %422 : tensor<128x4096x32xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x64xf32>, tensor<64xf32>) -> tensor<128x4096x32xf32>
    %154 = "mhlo.fusion"(%arg83, %arg82, %150, %145, %146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096x64xf32>, %arg212: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x4096x64xf32>
      %427 = stablehlo.transpose %426, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,64,4096]{1,2,0}"} : (tensor<64x4096x64xf32>) -> tensor<64x64x4096xf32>
      %428 = mhlo.copy %427 : tensor<64x64x4096xf32>
      %429 = mhlo.bitcast %428 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x64x4096xf32>) -> tensor<64x64x64x64xf32>
      %430 = stablehlo.transpose %429, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{2,1,3,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %431 = mhlo.copy %430 : tensor<64x64x64x64xf32>
      mhlo.return %431 : tensor<64x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x64x64x64xf32>
    %155 = "mhlo.fusion"(%arg90) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x4x4xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,64,64]{1,0,2,3}"} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
      %418 = mhlo.copy %417 : tensor<4x4x64x64xf32>
      mhlo.return %418 : tensor<4x4x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x4x4xf32>) -> tensor<4x4x64x64xf32>
    %156 = stablehlo.convolution(%154, %155) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x64x64x64xf32>, tensor<4x4x64x64xf32>) -> tensor<64x16x16x64xf32>
    %157 = "mhlo.fusion"(%156, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x64x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %423 = mhlo.copy %422 : tensor<64x256x64xf32>
      mhlo.return %423 : tensor<64x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x64xf32>, tensor<64xf32>) -> tensor<64x256x64xf32>
    %158 = "mhlo.fusion"(%157, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256xf32>
    %159 = "mhlo.fusion"(%158, %156, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x64xf32>, %arg210: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x64x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %423 = mhlo.copy %422 : tensor<64x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x64xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x64xf32>
      mhlo.return %428 : tensor<64x256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x64xf32>, tensor<64xf32>) -> tensor<64x256x64xf32>
    %160 = "mhlo.fusion"(%159, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256x2xf32>
      mhlo.return %417 : tensor<64x256x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x64xf32>, tensor<f32>) -> tensor<64x256x2xf32>
    %161 = "mhlo.fusion"(%160, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x2xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x2xf32>, tensor<f32>) -> tensor<64x256xf32>
    %162 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %163 = "mhlo.fusion"(%arg93, %arg92, %162, %158, %156, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x64xf32>, %arg213: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<64xf32>) -> tensor<64x16x16x64xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x64xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,16,16]{1,3,2,0}"} : (tensor<64x16x16x64xf32>) -> tensor<64x64x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x64x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x16x16xf32>) -> tensor<64x64x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,64]{1,2,0}"} : (tensor<64x64x256xf32>) -> tensor<64x256x64xf32>
      %423 = mhlo.copy %422 : tensor<64x256x64xf32>
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x64xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x64xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x64xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x64xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x256x64xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x64xf32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x64xf32>) -> tensor<16384x64xf32>
      mhlo.return %434 : tensor<16384x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x64xf32>, tensor<64xf32>) -> tensor<16384x64xf32>
    %164 = stablehlo.dot_general %163, %arg86, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x64xf32>) -> tensor<16384x64xf32>
    %165 = "mhlo.fusion"(%164, %arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,32,256]{2,1,3,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x32x256xf32>
      %421 = mhlo.copy %420 : tensor<64x2x32x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x32x256xf32>) -> tensor<128x32x256xf32>
      mhlo.return %422 : tensor<128x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>, tensor<64xf32>) -> tensor<128x32x256xf32>
    %166 = stablehlo.dot_general %153, %165, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x32xf32>, tensor<128x32x256xf32>) -> tensor<128x4096x256xf32>
    %167 = "mhlo.fusion"(%166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<128x4096x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
      mhlo.return %419 : tensor<64x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %168 = "mhlo.fusion"(%167, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x2x4096x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096x8xf32>
      mhlo.return %417 : tensor<64x2x4096x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096x8xf32>
    %169 = "mhlo.fusion"(%168, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x2x4096x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x4096x8xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
      mhlo.return %417 : tensor<64x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x8xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    %170 = "mhlo.fusion"(%169, %166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x2x4096xf32>, %arg209: tensor<128x4096x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128x4096x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<128x4096x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x2x4096xf32>
      %421 = stablehlo.maximum %420, %arg208 : tensor<64x2x4096xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<64x2x4096x256xf32>
      %424 = stablehlo.exponential %423 : tensor<64x2x4096x256xf32>
      mhlo.return %424 : tensor<64x2x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096xf32>, tensor<128x4096x256xf32>) -> tensor<64x2x4096x256xf32>
    %171 = "mhlo.fusion"(%170, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x2x4096x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
      mhlo.return %417 : tensor<64x2x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x256xf32>, tensor<f32>) -> tensor<64x2x4096xf32>
    %172 = "mhlo.fusion"(%170, %171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x2x4096x256xf32>, %arg209: tensor<64x2x4096xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<64x2x4096xf32>) -> tensor<64x2x4096x256xf32>
      %418 = stablehlo.divide %arg208, %417 : tensor<64x2x4096x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x4096x256xf32>) -> tensor<128x4096x256xf32>
      mhlo.return %419 : tensor<128x4096x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2x4096x256xf32>, tensor<64x2x4096xf32>) -> tensor<128x4096x256xf32>
    %173 = stablehlo.dot_general %163, %arg88, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x64xf32>, tensor<64x64xf32>) -> tensor<16384x64xf32>
    %174 = "mhlo.fusion"(%173, %arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x64xf32>, %arg209: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<64xf32>) -> tensor<16384x64xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x64xf32>) -> tensor<64x256x2x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,2,256,32]{3,1,2,0}"} : (tensor<64x256x2x32xf32>) -> tensor<64x2x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x2x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x2x256x32xf32>) -> tensor<128x256x32xf32>
      mhlo.return %422 : tensor<128x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x64xf32>, tensor<64xf32>) -> tensor<128x256x32xf32>
    %175 = stablehlo.dot_general %172, %174, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<128x4096x256xf32>, tensor<128x256x32xf32>) -> tensor<128x4096x32xf32>
    %176 = "mhlo.fusion"(%175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x4096x32xf32>):
      %417 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x4096x32xf32>) -> tensor<64x2x4096x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,4096,2,32]{3,1,2,0}"} : (tensor<64x2x4096x32xf32>) -> tensor<64x4096x2x32xf32>
      %419 = mhlo.copy %418 : tensor<64x4096x2x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x4096x2x32xf32>) -> tensor<262144x64xf32>
      mhlo.return %420 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128x4096x32xf32>) -> tensor<262144x64xf32>
    %177 = stablehlo.dot_general %176, %arg94, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<64x64xf32>) -> tensor<262144x64xf32>
    %178 = "mhlo.fusion"(%145, %177, %arg95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<262144x64xf32>, %arg210: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<262144x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x4096x64xf32>
      mhlo.return %420 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %179 = "mhlo.fusion"(%178, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %180 = "mhlo.fusion"(%178, %179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x4096x64xf32>
      mhlo.return %421 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %181 = "mhlo.fusion"(%180, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
      mhlo.return %417 : tensor<64x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    %182 = "mhlo.fusion"(%181, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x4096xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x4096xf32>
      mhlo.return %421 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>) -> tensor<64x4096xf32>
    %184 = "mhlo.fusion"(%arg97, %arg96, %183, %178, %179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096x64xf32>, %arg212: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x4096x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<262144x64xf32>
      mhlo.return %427 : tensor<262144x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<262144x64xf32>
    %185 = stablehlo.dot_general %184, %arg98, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x64xf32>, tensor<256x64xf32>) -> tensor<262144x256xf32>
    %186 = "mhlo.fusion"(%185, %arg99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<262144x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<262144x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<262144x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x256xf32>) -> tensor<64x4096x256xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,4096]{1,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
      %421 = mhlo.copy %420 : tensor<64x256x4096xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,256]{2,1,3,0}"} : (tensor<64x256x64x64xf32>) -> tensor<64x64x64x256xf32>
      %424 = mhlo.copy %423 : tensor<64x64x64x256xf32>
      mhlo.return %424 : tensor<64x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<262144x256xf32>, tensor<256xf32>) -> tensor<64x64x64x256xf32>
    %187 = "mhlo.fusion"(%arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x256xf32>
      mhlo.return %418 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %188 = stablehlo.convolution(%186, %187) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<64x64x64x256xf32>
    %189 = "mhlo.fusion"(%188, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x64x64x256xf32>, %arg209: tensor<256xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<64x64x64x256xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x64x64x256xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x64x64x256xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
      %422 = mhlo.copy %421 : tensor<64x256x64x64xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
      %425 = mhlo.copy %424 : tensor<64x4096x256xf32>
      %426 = stablehlo.negate %419 : tensor<64x64x64x256xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x64x64x256xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x64x64x256xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,64,64]{1,3,2,0}"} : (tensor<64x64x64x256xf32>) -> tensor<64x256x64x64xf32>
      %430 = mhlo.copy %429 : tensor<64x256x64x64xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x64x64xf32>) -> tensor<64x256x4096xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,4096,256]{1,2,0}"} : (tensor<64x256x4096xf32>) -> tensor<64x4096x256xf32>
      %433 = mhlo.copy %432 : tensor<64x4096x256xf32>
      %434 = stablehlo.abs %433 : tensor<64x4096x256xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x4096x256xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x4096x256xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x4096x256xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x4096x256xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x4096x256xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x4096x256xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x4096x256xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x4096x256xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x4096x256xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x4096x256xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x4096x256xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x4096x256xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x4096x256xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x4096x256xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x4096x256xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %462 = stablehlo.negate %438 : tensor<64x4096x256xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %465 = stablehlo.exponential %462 : tensor<64x4096x256xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x4096x256xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x4096x256xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x4096x256xf32>, tensor<64x4096x256xf32>) -> tensor<64x4096x256xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x4096x256xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x4096x256xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x4096x256xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x4096x256xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x4096x256xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x4096x256xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x4096x256xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x4096x256xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x4096x256xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x4096x256xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x4096x256xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x4096x256xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x4096x256xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x4096x256xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x4096x256xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x4096x256xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x4096x256xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x4096x256xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x4096x256xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x4096x256xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x4096x256xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x4096x256xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x4096x256xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x4096x256xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x4096x256xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x4096x256xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x4096x256xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x4096x256xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x4096x256xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x4096x256xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x4096x256xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x4096x256xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x4096x256xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x4096x256xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x4096x256xi1>, tensor<64x4096x256xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x4096x256xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x256xf32>) -> tensor<262144x256xf32>
      mhlo.return %524 : tensor<262144x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x64x256xf32>, tensor<256xf32>) -> tensor<262144x256xf32>
    %190 = stablehlo.dot_general %189, %arg102, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<262144x256xf32>, tensor<64x256xf32>) -> tensor<262144x64xf32>
    %191 = "mhlo.fusion"(%178, %190, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<262144x64xf32>, %arg210: tensor<64xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<64xf32>) -> tensor<262144x64xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<262144x64xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<262144x64xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x4096x64xf32>
      mhlo.return %420 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<262144x64xf32>, tensor<64xf32>) -> tensor<64x4096x64xf32>
    %192 = "mhlo.fusion"(%191, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %193 = "mhlo.fusion"(%191, %192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x4096x64xf32>
      mhlo.return %421 : tensor<64x4096x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
    %194 = "mhlo.fusion"(%193, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096x64xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
      mhlo.return %417 : tensor<64x4096x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x64xf32>, tensor<f32>) -> tensor<64x4096x2xf32>
    %195 = "mhlo.fusion"(%194, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x4096x2xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
      mhlo.return %417 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096x2xf32>, tensor<f32>) -> tensor<64x4096xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x4096xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x4096xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x4096xf32>
      mhlo.return %421 : tensor<64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64x4096xf32>) -> tensor<64x4096xf32>
    %197 = "mhlo.fusion"(%arg187, %arg186, %196, %191, %192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096x64xf32>, %arg212: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x4096x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<64x64x64x64xf32>
      %428 = stablehlo.transpose %427, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,64,64,64]{1,3,2,0}"} : (tensor<64x64x64x64xf32>) -> tensor<64x64x64x64xf32>
      %429 = mhlo.copy %428 : tensor<64x64x64x64xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x64x64x64xf32>) -> tensor<64x64x4096xf32>
      mhlo.return %430 : tensor<64x64x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x64x4096xf32>
    %198 = "mhlo.fusion"(%arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x64xf32>):
      %417 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[64,256]{0,1}"} : (tensor<256x64xf32>) -> tensor<64x256xf32>
      %418 = mhlo.copy %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [1, 2] : (tensor<64x256xf32>) -> tensor<64x64x256xf32>
      mhlo.return %419 : tensor<64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<64x64x256xf32>
    %199 = stablehlo.dot_general %197, %198, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x4096xf32>, tensor<64x64x256xf32>) -> tensor<64x4096x256xf32>
    %200 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<31> : tensor<i32>
      %417 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.iota dim = 0 : tensor<128xi32>
      %420 = stablehlo.convert %419 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %422 = stablehlo.add %420, %421 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %426 = stablehlo.add %424, %425 : tensor<128xf32>
      %427 = stablehlo.maximum %418, %426 : tensor<128xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %429 = stablehlo.convert %428 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %430 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %431 = stablehlo.add %429, %430 : tensor<128x1xi32>
      %432 = stablehlo.minimum %417, %431 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %433 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %434 = stablehlo.compare LT, %432, %433 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<32> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %436 = stablehlo.add %432, %435 : tensor<128x1xi32>
      %437 = stablehlo.select %434, %436, %432 : tensor<128x1xi1>, tensor<128x1xi32>
      %438 = mhlo.bitcast %437 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %440 = stablehlo.convert %427 : (tensor<128xf32>) -> tensor<128xi32>
      %441 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %442 = stablehlo.compare LT, %440, %441 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %443 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %444 = stablehlo.add %440, %443 : tensor<128xi32>
      %445 = stablehlo.select %442, %444, %440 : tensor<128xi1>, tensor<128xi32>
      %446 = stablehlo.broadcast_in_dim %445, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %447 = stablehlo.concatenate %439, %446, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %447 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %201 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.iota dim = 0 : tensor<128xi32>
      %419 = stablehlo.convert %418 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %425 = stablehlo.add %423, %424 : tensor<128xf32>
      %426 = stablehlo.maximum %417, %425 : tensor<128xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %428 = stablehlo.convert %427 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %429 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %430 = stablehlo.compare LT, %428, %429 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<32> : tensor<i32>
      %431 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %432 = stablehlo.add %428, %431 : tensor<128x1xi32>
      %433 = stablehlo.select %430, %432, %428 : tensor<128x1xi1>, tensor<128x1xi32>
      %434 = mhlo.bitcast %433 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %436 = stablehlo.convert %426 : (tensor<128xf32>) -> tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %439 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %440 = stablehlo.add %436, %439 : tensor<128xi32>
      %441 = stablehlo.select %438, %440, %436 : tensor<128xi1>, tensor<128xi32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %443 = stablehlo.concatenate %435, %442, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %443 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %202 = "mhlo.fusion"(%arg187, %arg186, %196, %191, %192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64xf32>, %arg209: tensor<64xf32>, %arg210: tensor<64x4096xf32>, %arg211: tensor<64x4096x64xf32>, %arg212: tensor<64x4096xf32>):
      %cst_2 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x4096xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x4096xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x4096x64xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x4096xf32>) -> tensor<64x4096x64xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x4096x64xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x4096x64xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<64xf32>) -> tensor<64x4096x64xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x4096x64xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x4096x64xf32>) -> tensor<64x64x64x64xf32>
      mhlo.return %427 : tensor<64x64x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64x4096xf32>, tensor<64x4096x64xf32>, tensor<64x4096xf32>) -> tensor<64x64x64x64xf32>
    %203 = "mhlo.fusion"(%arg8) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x64x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,64,160]{1,0,2,3}"} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
      %418 = mhlo.copy %417 : tensor<3x3x64x160xf32>
      mhlo.return %418 : tensor<3x3x64x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x64x3x3xf32>) -> tensor<3x3x64x160xf32>
    %204 = stablehlo.convolution(%202, %203) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x64x64x64xf32>, tensor<3x3x64x160xf32>) -> tensor<64x32x32x160xf32>
    %205 = "mhlo.fusion"(%204, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x32x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x32x32x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %420 = mhlo.copy %419 : tensor<64x160x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %423 = mhlo.copy %422 : tensor<64x1024x160xf32>
      mhlo.return %423 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x32x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %206 = "mhlo.fusion"(%205, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %207 = "mhlo.fusion"(%206, %204, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>, %arg209: tensor<64x32x32x160xf32>, %arg210: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x32x32x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %420 = mhlo.copy %419 : tensor<64x160x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %423 = mhlo.copy %422 : tensor<64x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x1024xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x1024x160xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x1024x160xf32>
      mhlo.return %428 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>, tensor<64x32x32x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %208 = "mhlo.fusion"(%207, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
      mhlo.return %417 : tensor<64x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    %209 = "mhlo.fusion"(%208, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %210 = "mhlo.fusion"(%209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x1024xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x1024xf32>
      mhlo.return %421 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %211 = "mhlo.fusion"(%arg11, %arg10, %210, %206, %204, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024xf32>, %arg212: tensor<64x32x32x160xf32>, %arg213: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x32x32x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %420 = mhlo.copy %419 : tensor<64x160x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %423 = mhlo.copy %422 : tensor<64x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x1024xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x1024x160xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x1024x160xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x1024x160xf32>
      mhlo.return %433 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<64x32x32x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %212 = "mhlo.fusion"(%211, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %213 = "mhlo.fusion"(%212, %arg11, %arg10, %210, %206, %204, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<64x1024xf32>, %arg212: tensor<64x1024xf32>, %arg213: tensor<64x32x32x160xf32>, %arg214: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %418 = stablehlo.add %arg213, %417 : tensor<64x32x32x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %420 = mhlo.copy %419 : tensor<64x160x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %423 = mhlo.copy %422 : tensor<64x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %425 = stablehlo.multiply %arg212, %424 : tensor<64x1024xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x1024x160xf32>
      %430 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x1024x160xf32>
      %432 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x1024x160xf32>
      %434 = stablehlo.multiply %arg208, %424 : tensor<64x1024xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x1024x160xf32>
      %437 = stablehlo.multiply %436, %436 : tensor<64x1024x160xf32>
      mhlo.return %437 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<64x32x32x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %214 = "mhlo.fusion"(%213, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
      mhlo.return %417 : tensor<64x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    %215 = "mhlo.fusion"(%214, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %216 = "mhlo.fusion"(%215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x1024xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x1024xf32>
      mhlo.return %421 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %217 = "mhlo.fusion"(%arg105, %arg104, %216, %212, %arg11, %arg10, %210, %206, %204, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<64x1024xf32>, %arg215: tensor<64x1024xf32>, %arg216: tensor<64x32x32x160xf32>, %arg217: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x32x32x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %420 = mhlo.copy %419 : tensor<64x160x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %423 = mhlo.copy %422 : tensor<64x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x1024xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x1024x160xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x1024x160xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x1024x160xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x1024xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x1024x160xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x1024x160xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x1024x160xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x1024x160xf32>
      %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %443 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<64x32x32x160xf32>, tensor<160xf32>) -> tensor<65536x160xf32>
    %218 = stablehlo.dot_general %217, %arg106, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %219 = "mhlo.fusion"(%218, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<65536x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,1024,32]{3,1,2,0}"} : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
      %421 = mhlo.copy %420 : tensor<64x5x1024x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x32xf32>) -> tensor<320x1024x32xf32>
      mhlo.return %422 : tensor<320x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<320x1024x32xf32>
    %220 = "mhlo.fusion"(%arg105, %arg104, %216, %212, %arg11, %arg10, %210, %206, %204, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024xf32>, %arg212: tensor<160xf32>, %arg213: tensor<160xf32>, %arg214: tensor<64x1024xf32>, %arg215: tensor<64x1024xf32>, %arg216: tensor<64x32x32x160xf32>, %arg217: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x32x32x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %420 = mhlo.copy %419 : tensor<64x160x32x32xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %423 = mhlo.copy %422 : tensor<64x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x1024xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x1024x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x1024x160xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x1024x160xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x1024x160xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x1024xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x1024x160xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x1024x160xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x1024x160xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x1024x160xf32>
      %443 = stablehlo.transpose %442, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,160,1024]{1,2,0}"} : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
      %444 = mhlo.copy %443 : tensor<64x160x1024xf32>
      %445 = mhlo.bitcast %444 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
      %446 = stablehlo.transpose %445, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,160]{2,1,3,0}"} : (tensor<64x160x32x32xf32>) -> tensor<64x32x32x160xf32>
      %447 = mhlo.copy %446 : tensor<64x32x32x160xf32>
      mhlo.return %447 : tensor<64x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<64x32x32x160xf32>, tensor<160xf32>) -> tensor<64x32x32x160xf32>
    %221 = "mhlo.fusion"(%arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %418 = mhlo.copy %417 : tensor<2x2x160x160xf32>
      mhlo.return %418 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %222 = stablehlo.convolution(%220, %221) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<64x16x16x160xf32>
    %223 = "mhlo.fusion"(%222, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x160x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %423 = mhlo.copy %422 : tensor<64x256x160xf32>
      mhlo.return %423 : tensor<64x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x160xf32>, tensor<160xf32>) -> tensor<64x256x160xf32>
    %224 = "mhlo.fusion"(%223, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256xf32>
    %225 = "mhlo.fusion"(%224, %222, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x160xf32>, %arg210: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x160x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %423 = mhlo.copy %422 : tensor<64x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x160xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x160xf32>
      mhlo.return %428 : tensor<64x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x160xf32>, tensor<160xf32>) -> tensor<64x256x160xf32>
    %226 = "mhlo.fusion"(%225, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256x5xf32>
      mhlo.return %417 : tensor<64x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256x5xf32>
    %227 = "mhlo.fusion"(%226, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x5xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x5xf32>, tensor<f32>) -> tensor<64x256xf32>
    %228 = "mhlo.fusion"(%227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %229 = "mhlo.fusion"(%arg115, %arg114, %228, %224, %222, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x160xf32>, %arg213: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x160x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %423 = mhlo.copy %422 : tensor<64x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x160xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x160xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x160xf32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
      mhlo.return %434 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x160xf32>, tensor<160xf32>) -> tensor<16384x160xf32>
    %230 = stablehlo.dot_general %229, %arg108, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %231 = "mhlo.fusion"(%230, %arg109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,32,256]{2,1,3,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
      %421 = mhlo.copy %420 : tensor<64x5x32x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x32x256xf32>) -> tensor<320x32x256xf32>
      mhlo.return %422 : tensor<320x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>) -> tensor<320x32x256xf32>
    %232 = stablehlo.dot_general %219, %231, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
    %233 = "mhlo.fusion"(%232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<320x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<320x1024x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
      mhlo.return %419 : tensor<64x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %234 = "mhlo.fusion"(%233, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x5x1024x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024x8xf32>
      mhlo.return %417 : tensor<64x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024x8xf32>
    %235 = "mhlo.fusion"(%234, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x5x1024x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<64x5x1024x8xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
      mhlo.return %417 : tensor<64x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x8xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    %236 = "mhlo.fusion"(%235, %232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x5x1024xf32>, %arg209: tensor<320x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<320x1024x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x5x1024xf32>
      %421 = stablehlo.maximum %420, %arg208 : tensor<64x5x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<64x5x1024x256xf32>
      %424 = stablehlo.exponential %423 : tensor<64x5x1024x256xf32>
      mhlo.return %424 : tensor<64x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024xf32>, tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %237 = "mhlo.fusion"(%236, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x5x1024x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
      mhlo.return %417 : tensor<64x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    %238 = "mhlo.fusion"(%236, %237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x5x1024x256xf32>, %arg209: tensor<64x5x1024xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %418 = stablehlo.divide %arg208, %417 : tensor<64x5x1024x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x256xf32>) -> tensor<320x1024x256xf32>
      mhlo.return %419 : tensor<320x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x256xf32>, tensor<64x5x1024xf32>) -> tensor<320x1024x256xf32>
    %239 = stablehlo.dot_general %229, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %240 = "mhlo.fusion"(%239, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,256,32]{3,1,2,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x5x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x256x32xf32>) -> tensor<320x256x32xf32>
      mhlo.return %422 : tensor<320x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>) -> tensor<320x256x32xf32>
    %241 = stablehlo.dot_general %238, %240, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
    %242 = "mhlo.fusion"(%241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<320x1024x32xf32>):
      %417 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,5,32]{3,1,2,0}"} : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
      %419 = mhlo.copy %418 : tensor<64x1024x5x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x5x32xf32>) -> tensor<65536x160xf32>
      mhlo.return %420 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x32xf32>) -> tensor<65536x160xf32>
    %243 = stablehlo.dot_general %242, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %244 = "mhlo.fusion"(%243, %arg11, %arg10, %210, %206, %204, %arg9, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<160xf32>, %arg211: tensor<64x1024xf32>, %arg212: tensor<64x1024xf32>, %arg213: tensor<64x32x32x160xf32>, %arg214: tensor<160xf32>, %arg215: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<65536x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<160xf32>) -> tensor<64x32x32x160xf32>
      %421 = stablehlo.add %arg213, %420 : tensor<64x32x32x160xf32>
      %422 = stablehlo.transpose %421, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %423 = mhlo.copy %422 : tensor<64x160x32x32xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,160]{1,2,0}"} : (tensor<64x160x1024xf32>) -> tensor<64x1024x160xf32>
      %426 = mhlo.copy %425 : tensor<64x1024x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %428 = stablehlo.multiply %arg212, %427 : tensor<64x1024xf32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %430 = stablehlo.subtract %426, %429 : tensor<64x1024x160xf32>
      %431 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<64x1024x160xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<64x1024x160xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %436 = stablehlo.add %434, %435 : tensor<64x1024x160xf32>
      %437 = stablehlo.add %419, %436 : tensor<64x1024x160xf32>
      mhlo.return %437 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024xf32>, tensor<64x32x32x160xf32>, tensor<160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %245 = "mhlo.fusion"(%244, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %246 = "mhlo.fusion"(%244, %245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x1024x160xf32>
      mhlo.return %421 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %247 = "mhlo.fusion"(%246, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
      mhlo.return %417 : tensor<64x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    %248 = "mhlo.fusion"(%247, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %249 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x1024xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x1024xf32>
      mhlo.return %421 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %250 = "mhlo.fusion"(%arg119, %arg118, %249, %244, %245) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024x160xf32>, %arg212: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x1024x160xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %427 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<65536x160xf32>
    %251 = stablehlo.dot_general %250, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
    %252 = "mhlo.fusion"(%251, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x640xf32>, %arg209: tensor<640xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<65536x640xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,640,1024]{1,2,0}"} : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
      %421 = mhlo.copy %420 : tensor<64x640x1024xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x640x1024xf32>) -> tensor<64x640x32x32xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,640]{2,1,3,0}"} : (tensor<64x640x32x32xf32>) -> tensor<64x32x32x640xf32>
      %424 = mhlo.copy %423 : tensor<64x32x32x640xf32>
      mhlo.return %424 : tensor<64x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x640xf32>, tensor<640xf32>) -> tensor<64x32x32x640xf32>
    %253 = "mhlo.fusion"(%arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x640xf32>
      mhlo.return %418 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %254 = stablehlo.convolution(%252, %253) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<64x32x32x640xf32>
    %255 = "mhlo.fusion"(%254, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<64x32x32x640xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x32x32x640xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x32x32x640xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
      %422 = mhlo.copy %421 : tensor<64x640x32x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
      %425 = mhlo.copy %424 : tensor<64x1024x640xf32>
      %426 = stablehlo.negate %419 : tensor<64x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x32x32x640xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
      %430 = mhlo.copy %429 : tensor<64x640x32x32xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
      %433 = mhlo.copy %432 : tensor<64x1024x640xf32>
      %434 = stablehlo.abs %433 : tensor<64x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x1024x640xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x1024x640xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x1024x640xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x1024x640xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x1024x640xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x1024x640xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x1024x640xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x1024x640xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %462 = stablehlo.negate %438 : tensor<64x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %465 = stablehlo.exponential %462 : tensor<64x1024x640xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x1024x640xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x1024x640xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x1024x640xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x1024x640xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x1024x640xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x1024x640xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x1024x640xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x1024x640xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x1024x640xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x1024x640xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x1024x640xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x1024x640xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x1024x640xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x1024x640xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x1024x640xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x1024x640xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x1024x640xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x1024x640xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x1024x640xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x1024x640xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
      mhlo.return %524 : tensor<65536x640xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x32x640xf32>, tensor<640xf32>) -> tensor<65536x640xf32>
    %256 = stablehlo.dot_general %255, %arg124, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x640xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
    %257 = "mhlo.fusion"(%244, %256, %arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<65536x160xf32>, %arg210: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<65536x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x1024x160xf32>
      mhlo.return %420 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %258 = "mhlo.fusion"(%257, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %259 = "mhlo.fusion"(%257, %258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x1024x160xf32>
      mhlo.return %421 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %260 = "mhlo.fusion"(%259, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
      mhlo.return %417 : tensor<64x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    %261 = "mhlo.fusion"(%260, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %262 = "mhlo.fusion"(%261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x1024xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x1024xf32>
      mhlo.return %421 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %263 = "mhlo.fusion"(%arg127, %arg126, %262, %257, %258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024x160xf32>, %arg212: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x1024x160xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %427 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<65536x160xf32>
    %264 = stablehlo.dot_general %263, %arg128, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %265 = "mhlo.fusion"(%264, %arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<65536x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x5x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,1024,32]{3,1,2,0}"} : (tensor<64x1024x5x32xf32>) -> tensor<64x5x1024x32xf32>
      %421 = mhlo.copy %420 : tensor<64x5x1024x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x32xf32>) -> tensor<320x1024x32xf32>
      mhlo.return %422 : tensor<320x1024x32xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x160xf32>, tensor<160xf32>) -> tensor<320x1024x32xf32>
    %266 = "mhlo.fusion"(%arg127, %arg126, %262, %257, %258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024x160xf32>, %arg212: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x1024x160xf32>
      %427 = stablehlo.transpose %426, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,160,1024]{1,2,0}"} : (tensor<64x1024x160xf32>) -> tensor<64x160x1024xf32>
      %428 = mhlo.copy %427 : tensor<64x160x1024xf32>
      %429 = mhlo.bitcast %428 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x160x1024xf32>) -> tensor<64x160x32x32xf32>
      %430 = stablehlo.transpose %429, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,160]{2,1,3,0}"} : (tensor<64x160x32x32xf32>) -> tensor<64x32x32x160xf32>
      %431 = mhlo.copy %430 : tensor<64x32x32x160xf32>
      mhlo.return %431 : tensor<64x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x32x32x160xf32>
    %267 = "mhlo.fusion"(%arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160x160x2x2xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,160,160]{1,0,2,3}"} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
      %418 = mhlo.copy %417 : tensor<2x2x160x160xf32>
      mhlo.return %418 : tensor<2x2x160x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160x160x2x2xf32>) -> tensor<2x2x160x160xf32>
    %268 = stablehlo.convolution(%266, %267) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x32x32x160xf32>, tensor<2x2x160x160xf32>) -> tensor<64x16x16x160xf32>
    %269 = "mhlo.fusion"(%268, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x160x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %423 = mhlo.copy %422 : tensor<64x256x160xf32>
      mhlo.return %423 : tensor<64x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x160xf32>, tensor<160xf32>) -> tensor<64x256x160xf32>
    %270 = "mhlo.fusion"(%269, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256xf32>
    %271 = "mhlo.fusion"(%270, %268, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x160xf32>, %arg210: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x160x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %423 = mhlo.copy %422 : tensor<64x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x160xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x160xf32>
      mhlo.return %428 : tensor<64x256x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x160xf32>, tensor<160xf32>) -> tensor<64x256x160xf32>
    %272 = "mhlo.fusion"(%271, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256x5xf32>
      mhlo.return %417 : tensor<64x256x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x160xf32>, tensor<f32>) -> tensor<64x256x5xf32>
    %273 = "mhlo.fusion"(%272, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x5xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x5xf32>, tensor<f32>) -> tensor<64x256xf32>
    %274 = "mhlo.fusion"(%273) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %275 = "mhlo.fusion"(%arg137, %arg136, %274, %270, %268, %arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x160xf32>, %arg213: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<160xf32>) -> tensor<64x16x16x160xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x160xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,16,16]{1,3,2,0}"} : (tensor<64x16x16x160xf32>) -> tensor<64x160x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x160x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x16x16xf32>) -> tensor<64x160x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,160]{1,2,0}"} : (tensor<64x160x256xf32>) -> tensor<64x256x160xf32>
      %423 = mhlo.copy %422 : tensor<64x256x160xf32>
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x160xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x160xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x160xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x160xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x256x160xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x160xf32>
      %434 = mhlo.bitcast %433 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x160xf32>) -> tensor<16384x160xf32>
      mhlo.return %434 : tensor<16384x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x160xf32>, tensor<160xf32>) -> tensor<16384x160xf32>
    %276 = stablehlo.dot_general %275, %arg130, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %277 = "mhlo.fusion"(%276, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,32,256]{2,1,3,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x32x256xf32>
      %421 = mhlo.copy %420 : tensor<64x5x32x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x32x256xf32>) -> tensor<320x32x256xf32>
      mhlo.return %422 : tensor<320x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>) -> tensor<320x32x256xf32>
    %278 = stablehlo.dot_general %265, %277, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x32xf32>, tensor<320x32x256xf32>) -> tensor<320x1024x256xf32>
    %279 = "mhlo.fusion"(%278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<320x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<320x1024x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
      mhlo.return %419 : tensor<64x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %280 = "mhlo.fusion"(%279, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x5x1024x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024x8xf32>
      mhlo.return %417 : tensor<64x5x1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024x8xf32>
    %281 = "mhlo.fusion"(%280, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x5x1024x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<64x5x1024x8xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
      mhlo.return %417 : tensor<64x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x8xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    %282 = "mhlo.fusion"(%281, %278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x5x1024xf32>, %arg209: tensor<320x1024x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<320x1024x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<320x1024x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x5x1024xf32>
      %421 = stablehlo.maximum %420, %arg208 : tensor<64x5x1024xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<64x5x1024x256xf32>
      %424 = stablehlo.exponential %423 : tensor<64x5x1024x256xf32>
      mhlo.return %424 : tensor<64x5x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024xf32>, tensor<320x1024x256xf32>) -> tensor<64x5x1024x256xf32>
    %283 = "mhlo.fusion"(%282, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x5x1024x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
      mhlo.return %417 : tensor<64x5x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x256xf32>, tensor<f32>) -> tensor<64x5x1024xf32>
    %284 = "mhlo.fusion"(%282, %283) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x5x1024x256xf32>, %arg209: tensor<64x5x1024xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<64x5x1024xf32>) -> tensor<64x5x1024x256xf32>
      %418 = stablehlo.divide %arg208, %417 : tensor<64x5x1024x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x1024x256xf32>) -> tensor<320x1024x256xf32>
      mhlo.return %419 : tensor<320x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x5x1024x256xf32>, tensor<64x5x1024xf32>) -> tensor<320x1024x256xf32>
    %285 = stablehlo.dot_general %275, %arg132, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x160xf32>, tensor<160x160xf32>) -> tensor<16384x160xf32>
    %286 = "mhlo.fusion"(%285, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x160xf32>, %arg209: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<160xf32>) -> tensor<16384x160xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x160xf32>) -> tensor<64x256x5x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,5,256,32]{3,1,2,0}"} : (tensor<64x256x5x32xf32>) -> tensor<64x5x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x5x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x5x256x32xf32>) -> tensor<320x256x32xf32>
      mhlo.return %422 : tensor<320x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x160xf32>, tensor<160xf32>) -> tensor<320x256x32xf32>
    %287 = stablehlo.dot_general %284, %286, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<320x1024x256xf32>, tensor<320x256x32xf32>) -> tensor<320x1024x32xf32>
    %288 = "mhlo.fusion"(%287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<320x1024x32xf32>):
      %417 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<320x1024x32xf32>) -> tensor<64x5x1024x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,5,32]{3,1,2,0}"} : (tensor<64x5x1024x32xf32>) -> tensor<64x1024x5x32xf32>
      %419 = mhlo.copy %418 : tensor<64x1024x5x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x5x32xf32>) -> tensor<65536x160xf32>
      mhlo.return %420 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<320x1024x32xf32>) -> tensor<65536x160xf32>
    %289 = stablehlo.dot_general %288, %arg138, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<160x160xf32>) -> tensor<65536x160xf32>
    %290 = "mhlo.fusion"(%257, %289, %arg139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<65536x160xf32>, %arg210: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<65536x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x1024x160xf32>
      mhlo.return %420 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %291 = "mhlo.fusion"(%290, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %292 = "mhlo.fusion"(%290, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x1024x160xf32>
      mhlo.return %421 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %293 = "mhlo.fusion"(%292, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
      mhlo.return %417 : tensor<64x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    %294 = "mhlo.fusion"(%293, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %295 = "mhlo.fusion"(%294) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x1024xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x1024xf32>
      mhlo.return %421 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %296 = "mhlo.fusion"(%arg141, %arg140, %295, %290, %291) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024x160xf32>, %arg212: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x1024x160xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<65536x160xf32>
      mhlo.return %427 : tensor<65536x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<65536x160xf32>
    %297 = stablehlo.dot_general %296, %arg142, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x160xf32>, tensor<640x160xf32>) -> tensor<65536x640xf32>
    %298 = "mhlo.fusion"(%297, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<65536x640xf32>, %arg209: tensor<640xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<640xf32>) -> tensor<65536x640xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<65536x640xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x640xf32>) -> tensor<64x1024x640xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,640,1024]{1,2,0}"} : (tensor<64x1024x640xf32>) -> tensor<64x640x1024xf32>
      %421 = mhlo.copy %420 : tensor<64x640x1024xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x640x1024xf32>) -> tensor<64x640x32x32xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,32,32,640]{2,1,3,0}"} : (tensor<64x640x32x32xf32>) -> tensor<64x32x32x640xf32>
      %424 = mhlo.copy %423 : tensor<64x32x32x640xf32>
      mhlo.return %424 : tensor<64x32x32x640xf32>
    }) {output_operand_aliasing = []} : (tensor<65536x640xf32>, tensor<640xf32>) -> tensor<64x32x32x640xf32>
    %299 = "mhlo.fusion"(%arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<640x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,640]{1,0,2,3}"} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x640xf32>
      mhlo.return %418 : tensor<3x3x1x640xf32>
    }) {output_operand_aliasing = []} : (tensor<640x1x3x3xf32>) -> tensor<3x3x1x640xf32>
    %300 = stablehlo.convolution(%298, %299) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 640 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x32x32x640xf32>, tensor<3x3x1x640xf32>) -> tensor<64x32x32x640xf32>
    %301 = "mhlo.fusion"(%300, %arg145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x32x32x640xf32>, %arg209: tensor<640xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<640xf32>) -> tensor<64x32x32x640xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x32x32x640xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x32x32x640xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
      %422 = mhlo.copy %421 : tensor<64x640x32x32xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
      %425 = mhlo.copy %424 : tensor<64x1024x640xf32>
      %426 = stablehlo.negate %419 : tensor<64x32x32x640xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x32x32x640xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x32x32x640xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,640,32,32]{1,3,2,0}"} : (tensor<64x32x32x640xf32>) -> tensor<64x640x32x32xf32>
      %430 = mhlo.copy %429 : tensor<64x640x32x32xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x640x32x32xf32>) -> tensor<64x640x1024xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,640]{1,2,0}"} : (tensor<64x640x1024xf32>) -> tensor<64x1024x640xf32>
      %433 = mhlo.copy %432 : tensor<64x1024x640xf32>
      %434 = stablehlo.abs %433 : tensor<64x1024x640xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x1024x640xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x1024x640xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x1024x640xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x1024x640xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x1024x640xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x1024x640xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x1024x640xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x1024x640xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x1024x640xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x1024x640xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x1024x640xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x1024x640xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x1024x640xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x1024x640xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x1024x640xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %462 = stablehlo.negate %438 : tensor<64x1024x640xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %465 = stablehlo.exponential %462 : tensor<64x1024x640xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x1024x640xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x1024x640xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x1024x640xf32>, tensor<64x1024x640xf32>) -> tensor<64x1024x640xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x1024x640xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x1024x640xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x1024x640xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x1024x640xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x1024x640xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x1024x640xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x1024x640xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x1024x640xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x1024x640xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x1024x640xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x1024x640xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x1024x640xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x1024x640xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x1024x640xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x1024x640xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x1024x640xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x1024x640xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x1024x640xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x1024x640xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x1024x640xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x1024x640xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x1024x640xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x1024x640xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x1024x640xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x1024x640xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x1024x640xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x1024x640xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x1024x640xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x1024x640xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x1024x640xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x1024x640xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x1024x640xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x1024x640xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x1024x640xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x1024x640xi1>, tensor<64x1024x640xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x1024x640xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x640xf32>) -> tensor<65536x640xf32>
      mhlo.return %524 : tensor<65536x640xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x32x640xf32>, tensor<640xf32>) -> tensor<65536x640xf32>
    %302 = stablehlo.dot_general %301, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<65536x640xf32>, tensor<160x640xf32>) -> tensor<65536x160xf32>
    %303 = "mhlo.fusion"(%290, %302, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<65536x160xf32>, %arg210: tensor<160xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<160xf32>) -> tensor<65536x160xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<65536x160xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<65536x160xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x1024x160xf32>
      mhlo.return %420 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<65536x160xf32>, tensor<160xf32>) -> tensor<64x1024x160xf32>
    %304 = "mhlo.fusion"(%303, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %305 = "mhlo.fusion"(%303, %304) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x1024x160xf32>
      mhlo.return %421 : tensor<64x1024x160xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
    %306 = "mhlo.fusion"(%305, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024x160xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
      mhlo.return %417 : tensor<64x1024x5xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x160xf32>, tensor<f32>) -> tensor<64x1024x5xf32>
    %307 = "mhlo.fusion"(%306, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x1024x5xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %417 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024x5xf32>, tensor<f32>) -> tensor<64x1024xf32>
    %308 = "mhlo.fusion"(%307) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x1024xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x1024xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x1024xf32>
      mhlo.return %421 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<64x1024xf32>
    %309 = "mhlo.fusion"(%arg189, %arg188, %308, %303, %304) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024x160xf32>, %arg212: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x1024x160xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<64x32x32x160xf32>
      %428 = stablehlo.transpose %427, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,160,32,32]{1,3,2,0}"} : (tensor<64x32x32x160xf32>) -> tensor<64x160x32x32xf32>
      %429 = mhlo.copy %428 : tensor<64x160x32x32xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x160x32x32xf32>) -> tensor<64x160x1024xf32>
      mhlo.return %430 : tensor<64x160x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x160x1024xf32>
    %310 = "mhlo.fusion"(%arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160xf32>):
      %417 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[160,256]{0,1}"} : (tensor<256x160xf32>) -> tensor<160x256xf32>
      %418 = mhlo.copy %417 : tensor<160x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [1, 2] : (tensor<160x256xf32>) -> tensor<64x160x256xf32>
      mhlo.return %419 : tensor<64x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160xf32>) -> tensor<64x160x256xf32>
    %311 = stablehlo.dot_general %309, %310, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x160x1024xf32>, tensor<64x160x256xf32>) -> tensor<64x1024x256xf32>
    %312 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %c_2 = stablehlo.constant dense<15> : tensor<i32>
      %417 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %418 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %419 = stablehlo.iota dim = 0 : tensor<128xi32>
      %420 = stablehlo.convert %419 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %421 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %422 = stablehlo.add %420, %421 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %423 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %426 = stablehlo.add %424, %425 : tensor<128xf32>
      %427 = stablehlo.maximum %418, %426 : tensor<128xf32>
      %428 = mhlo.bitcast %427 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %429 = stablehlo.convert %428 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_7 = stablehlo.constant dense<1> : tensor<i32>
      %430 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %431 = stablehlo.add %429, %430 : tensor<128x1xi32>
      %432 = stablehlo.minimum %417, %431 : tensor<128x1xi32>
      %c_8 = stablehlo.constant dense<0> : tensor<i32>
      %433 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %434 = stablehlo.compare LT, %432, %433 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_9 = stablehlo.constant dense<16> : tensor<i32>
      %435 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %436 = stablehlo.add %432, %435 : tensor<128x1xi32>
      %437 = stablehlo.select %434, %436, %432 : tensor<128x1xi1>, tensor<128x1xi32>
      %438 = mhlo.bitcast %437 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %439 = stablehlo.broadcast_in_dim %438, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %440 = stablehlo.convert %427 : (tensor<128xf32>) -> tensor<128xi32>
      %441 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %442 = stablehlo.compare LT, %440, %441 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %443 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %444 = stablehlo.add %440, %443 : tensor<128xi32>
      %445 = stablehlo.select %442, %444, %440 : tensor<128xi1>, tensor<128xi32>
      %446 = stablehlo.broadcast_in_dim %445, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %447 = stablehlo.concatenate %439, %446, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %447 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %313 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %418 = stablehlo.iota dim = 0 : tensor<128xi32>
      %419 = stablehlo.convert %418 : (tensor<128xi32>) -> tensor<128xf32>
      %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %421 = stablehlo.add %419, %420 : tensor<128xf32>
      %cst_4 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %423 = stablehlo.multiply %421, %422 : tensor<128xf32>
      %cst_5 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %425 = stablehlo.add %423, %424 : tensor<128xf32>
      %426 = stablehlo.maximum %417, %425 : tensor<128xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %428 = stablehlo.convert %427 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %c_6 = stablehlo.constant dense<0> : tensor<i32>
      %429 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %430 = stablehlo.compare LT, %428, %429 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_7 = stablehlo.constant dense<16> : tensor<i32>
      %431 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %432 = stablehlo.add %428, %431 : tensor<128x1xi32>
      %433 = stablehlo.select %430, %432, %428 : tensor<128x1xi1>, tensor<128x1xi32>
      %434 = mhlo.bitcast %433 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %436 = stablehlo.convert %426 : (tensor<128xf32>) -> tensor<128xi32>
      %437 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %438 = stablehlo.compare LT, %436, %437 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %439 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %440 = stablehlo.add %436, %439 : tensor<128xi32>
      %441 = stablehlo.select %438, %440, %436 : tensor<128xi1>, tensor<128xi32>
      %442 = stablehlo.broadcast_in_dim %441, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %443 = stablehlo.concatenate %435, %442, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      mhlo.return %443 : tensor<128x128x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<128x128x2xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_1 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %314 = "mhlo.fusion"(%arg189, %arg188, %308, %303, %304) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<160xf32>, %arg209: tensor<160xf32>, %arg210: tensor<64x1024xf32>, %arg211: tensor<64x1024x160xf32>, %arg212: tensor<64x1024xf32>):
      %cst_2 = stablehlo.constant dense<6.250000e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x1024xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x1024xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x1024x160xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x1024xf32>) -> tensor<64x1024x160xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x1024x160xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x1024x160xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<160xf32>) -> tensor<64x1024x160xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x1024x160xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x160xf32>) -> tensor<64x32x32x160xf32>
      mhlo.return %427 : tensor<64x32x32x160xf32>
    }) {output_operand_aliasing = []} : (tensor<160xf32>, tensor<160xf32>, tensor<64x1024xf32>, tensor<64x1024x160xf32>, tensor<64x1024xf32>) -> tensor<64x32x32x160xf32>
    %315 = "mhlo.fusion"(%arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x160x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,160,256]{1,0,2,3}"} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
      %418 = mhlo.copy %417 : tensor<3x3x160x256xf32>
      mhlo.return %418 : tensor<3x3x160x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x160x3x3xf32>) -> tensor<3x3x160x256xf32>
    %316 = stablehlo.convolution(%314, %315) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x32x32x160xf32>, tensor<3x3x160x256xf32>) -> tensor<64x16x16x256xf32>
    %317 = "mhlo.fusion"(%316, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x16x16x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x256x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %423 = mhlo.copy %422 : tensor<64x256x256xf32>
      mhlo.return %423 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %318 = "mhlo.fusion"(%317, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %319 = "mhlo.fusion"(%318, %316, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<64x16x16x256xf32>, %arg210: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
      %418 = stablehlo.add %arg209, %417 : tensor<64x16x16x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x256x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %423 = mhlo.copy %422 : tensor<64x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x256xf32>
      %428 = stablehlo.multiply %427, %427 : tensor<64x256x256xf32>
      mhlo.return %428 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<64x16x16x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %320 = "mhlo.fusion"(%319, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
      mhlo.return %417 : tensor<64x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    %321 = "mhlo.fusion"(%320, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    %322 = "mhlo.fusion"(%321) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %323 = "mhlo.fusion"(%arg15, %arg14, %322, %318, %316, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x16x16x256xf32>, %arg213: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg213, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
      %418 = stablehlo.add %arg212, %417 : tensor<64x16x16x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x256x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %423 = mhlo.copy %422 : tensor<64x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x256xf32>
      %428 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x256xf32>
      %430 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x256xf32>
      %432 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x256xf32>
      mhlo.return %433 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %324 = "mhlo.fusion"(%323, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %325 = "mhlo.fusion"(%324, %arg15, %arg14, %322, %318, %316, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x256xf32>, %arg213: tensor<64x16x16x256xf32>, %arg214: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
      %418 = stablehlo.add %arg213, %417 : tensor<64x16x16x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x256x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %423 = mhlo.copy %422 : tensor<64x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg212, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x256xf32>
      %428 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x256xf32>
      %430 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x256xf32>
      %432 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x256xf32>
      %434 = stablehlo.multiply %arg208, %424 : tensor<64x256xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x256x256xf32>
      %437 = stablehlo.multiply %436, %436 : tensor<64x256x256xf32>
      mhlo.return %437 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %326 = "mhlo.fusion"(%325, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
      mhlo.return %417 : tensor<64x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    %327 = "mhlo.fusion"(%326, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    %328 = "mhlo.fusion"(%327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %329 = "mhlo.fusion"(%arg149, %arg148, %328, %324, %arg15, %arg14, %322, %318, %316, %arg13) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<256xf32>, %arg213: tensor<256xf32>, %arg214: tensor<64x256xf32>, %arg215: tensor<64x256xf32>, %arg216: tensor<64x16x16x256xf32>, %arg217: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg217, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
      %418 = stablehlo.add %arg216, %417 : tensor<64x16x16x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %420 = mhlo.copy %419 : tensor<64x256x16x16xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %423 = mhlo.copy %422 : tensor<64x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %424 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %425 = stablehlo.multiply %arg215, %424 : tensor<64x256xf32>
      %426 = stablehlo.broadcast_in_dim %425, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %427 = stablehlo.subtract %423, %426 : tensor<64x256x256xf32>
      %428 = stablehlo.broadcast_in_dim %arg214, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %429 = stablehlo.multiply %427, %428 : tensor<64x256x256xf32>
      %430 = stablehlo.broadcast_in_dim %arg213, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %431 = stablehlo.multiply %429, %430 : tensor<64x256x256xf32>
      %432 = stablehlo.broadcast_in_dim %arg212, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %433 = stablehlo.add %431, %432 : tensor<64x256x256xf32>
      %434 = stablehlo.multiply %arg211, %424 : tensor<64x256xf32>
      %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %436 = stablehlo.subtract %433, %435 : tensor<64x256x256xf32>
      %437 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %438 = stablehlo.multiply %436, %437 : tensor<64x256x256xf32>
      %439 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<64x256x256xf32>
      %441 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %442 = stablehlo.add %440, %441 : tensor<64x256x256xf32>
      %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %443 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x256xf32>, tensor<256xf32>) -> tensor<16384x256xf32>
    %330 = stablehlo.dot_general %329, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %331 = "mhlo.fusion"(%330, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x8x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
      mhlo.return %422 : tensor<512x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>) -> tensor<512x256x32xf32>
    %332 = stablehlo.dot_general %329, %arg152, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %333 = "mhlo.fusion"(%332, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,32,256]{2,1,3,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
      %421 = mhlo.copy %420 : tensor<64x8x32x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x32x256xf32>) -> tensor<512x32x256xf32>
      mhlo.return %422 : tensor<512x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>) -> tensor<512x32x256xf32>
    %334 = stablehlo.dot_general %331, %333, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
    %335 = "mhlo.fusion"(%334) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<512x256x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
      mhlo.return %419 : tensor<64x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %336 = "mhlo.fusion"(%335, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x8x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256x8xf32>
      mhlo.return %417 : tensor<64x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256x8xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x8x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x256x8xf32>, tensor<f32>) -> tensor<64x8x256xf32>
      mhlo.return %417 : tensor<64x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x8xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    %338 = "mhlo.fusion"(%337, %334) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x8x256xf32>, %arg209: tensor<512x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<512x256x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x8x256xf32>
      %421 = stablehlo.maximum %420, %arg208 : tensor<64x8x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<64x8x256x256xf32>
      %424 = stablehlo.exponential %423 : tensor<64x8x256x256xf32>
      mhlo.return %424 : tensor<64x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256xf32>, tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %339 = "mhlo.fusion"(%338, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x8x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
      mhlo.return %417 : tensor<64x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    %340 = "mhlo.fusion"(%338, %339) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x8x256x256xf32>, %arg209: tensor<64x8x256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %418 = stablehlo.divide %arg208, %417 : tensor<64x8x256x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x256xf32>) -> tensor<512x256x256xf32>
      mhlo.return %419 : tensor<512x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x256xf32>, tensor<64x8x256xf32>) -> tensor<512x256x256xf32>
    %341 = stablehlo.dot_general %329, %arg154, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %342 = "mhlo.fusion"(%341, %arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x8x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
      mhlo.return %422 : tensor<512x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>) -> tensor<512x256x32xf32>
    %343 = stablehlo.dot_general %340, %342, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
    %344 = "mhlo.fusion"(%343) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x256x32xf32>):
      %417 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,8,32]{3,1,2,0}"} : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
      %419 = mhlo.copy %418 : tensor<64x256x8x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x8x32xf32>) -> tensor<16384x256xf32>
      mhlo.return %420 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x32xf32>) -> tensor<16384x256xf32>
    %345 = stablehlo.dot_general %344, %arg156, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %346 = "mhlo.fusion"(%345, %arg15, %arg14, %322, %318, %316, %arg13, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<64x256xf32>, %arg212: tensor<64x256xf32>, %arg213: tensor<64x16x16x256xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg215, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.broadcast_in_dim %arg214, dims = [3] : (tensor<256xf32>) -> tensor<64x16x16x256xf32>
      %421 = stablehlo.add %arg213, %420 : tensor<64x16x16x256xf32>
      %422 = stablehlo.transpose %421, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %423 = mhlo.copy %422 : tensor<64x256x16x16xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      %425 = stablehlo.transpose %424, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %426 = mhlo.copy %425 : tensor<64x256x256xf32>
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %428 = stablehlo.multiply %arg212, %427 : tensor<64x256xf32>
      %429 = stablehlo.broadcast_in_dim %428, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %430 = stablehlo.subtract %426, %429 : tensor<64x256x256xf32>
      %431 = stablehlo.broadcast_in_dim %arg211, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %432 = stablehlo.multiply %430, %431 : tensor<64x256x256xf32>
      %433 = stablehlo.broadcast_in_dim %arg210, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %434 = stablehlo.multiply %432, %433 : tensor<64x256x256xf32>
      %435 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %436 = stablehlo.add %434, %435 : tensor<64x256x256xf32>
      %437 = stablehlo.add %419, %436 : tensor<64x256x256xf32>
      mhlo.return %437 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<64x16x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %347 = "mhlo.fusion"(%346, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %348 = "mhlo.fusion"(%346, %347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x256x256xf32>
      mhlo.return %421 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %349 = "mhlo.fusion"(%348, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
      mhlo.return %417 : tensor<64x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    %350 = "mhlo.fusion"(%349, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    %351 = "mhlo.fusion"(%350) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %352 = "mhlo.fusion"(%arg159, %arg158, %351, %346, %347) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256x256xf32>, %arg212: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x256x256xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x256x256xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x256x256xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %427 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<16384x256xf32>
    %353 = stablehlo.dot_general %352, %arg160, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
    %354 = "mhlo.fusion"(%353, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x1024xf32>, %arg209: tensor<1024xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x1024xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,256]{1,2,0}"} : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
      %421 = mhlo.copy %420 : tensor<64x1024x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x256xf32>) -> tensor<64x1024x16x16xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,16,16,1024]{2,1,3,0}"} : (tensor<64x1024x16x16xf32>) -> tensor<64x16x16x1024xf32>
      %424 = mhlo.copy %423 : tensor<64x16x16x1024xf32>
      mhlo.return %424 : tensor<64x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x1024xf32>, tensor<1024xf32>) -> tensor<64x16x16x1024xf32>
    %355 = "mhlo.fusion"(%arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x1024xf32>
      mhlo.return %418 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %356 = stablehlo.convolution(%354, %355) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<64x16x16x1024xf32>
    %357 = "mhlo.fusion"(%356, %arg163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<64x16x16x1024xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x16x16x1024xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x16x16x1024xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
      %422 = mhlo.copy %421 : tensor<64x1024x16x16xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %425 = mhlo.copy %424 : tensor<64x256x1024xf32>
      %426 = stablehlo.negate %419 : tensor<64x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x16x16x1024xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
      %430 = mhlo.copy %429 : tensor<64x1024x16x16xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %433 = mhlo.copy %432 : tensor<64x256x1024xf32>
      %434 = stablehlo.abs %433 : tensor<64x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x256x1024xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x256x1024xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x256x1024xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x256x1024xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x256x1024xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x256x1024xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x256x1024xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x256x1024xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %462 = stablehlo.negate %438 : tensor<64x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %465 = stablehlo.exponential %462 : tensor<64x256x1024xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x256x1024xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x256x1024xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x256x1024xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x256x1024xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x256x1024xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x256x1024xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x256x1024xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x256x1024xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x256x1024xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x256x1024xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x256x1024xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x256x1024xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x256x1024xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x256x1024xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x256x1024xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x256x1024xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x256x1024xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x256x1024xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x256x1024xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x256x1024xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
      mhlo.return %524 : tensor<16384x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x1024xf32>, tensor<1024xf32>) -> tensor<16384x1024xf32>
    %358 = stablehlo.dot_general %357, %arg164, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
    %359 = "mhlo.fusion"(%346, %358, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<16384x256xf32>, %arg210: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x256x256xf32>
      mhlo.return %420 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %360 = "mhlo.fusion"(%359, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %361 = "mhlo.fusion"(%359, %360) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x256x256xf32>
      mhlo.return %421 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %362 = "mhlo.fusion"(%361, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
      mhlo.return %417 : tensor<64x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    %363 = "mhlo.fusion"(%362, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    %364 = "mhlo.fusion"(%363) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %365 = "mhlo.fusion"(%arg167, %arg166, %364, %359, %360) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256x256xf32>, %arg212: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x256x256xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x256x256xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x256x256xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %427 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<16384x256xf32>
    %366 = stablehlo.dot_general %365, %arg168, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %367 = "mhlo.fusion"(%366, %arg169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x8x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
      mhlo.return %422 : tensor<512x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>) -> tensor<512x256x32xf32>
    %368 = stablehlo.dot_general %365, %arg170, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %369 = "mhlo.fusion"(%368, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,32,256]{2,1,3,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x32x256xf32>
      %421 = mhlo.copy %420 : tensor<64x8x32x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x32x256xf32>) -> tensor<512x32x256xf32>
      mhlo.return %422 : tensor<512x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>) -> tensor<512x32x256xf32>
    %370 = stablehlo.dot_general %367, %369, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x32xf32>, tensor<512x32x256xf32>) -> tensor<512x256x256xf32>
    %371 = "mhlo.fusion"(%370) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<512x256x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
      mhlo.return %419 : tensor<64x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %372 = "mhlo.fusion"(%371, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x8x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.maximum %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256x8xf32>
      mhlo.return %417 : tensor<64x8x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256x8xf32>
    %373 = "mhlo.fusion"(%372, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x8x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.maximum across dimensions = [3] : (tensor<64x8x256x8xf32>, tensor<f32>) -> tensor<64x8x256xf32>
      mhlo.return %417 : tensor<64x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x8xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    %374 = "mhlo.fusion"(%373, %370) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x8x256xf32>, %arg209: tensor<512x256x256xf32>):
      %cst_2 = stablehlo.constant dense<0.176776692> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x256x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<512x256x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
      %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %420 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x8x256xf32>
      %421 = stablehlo.maximum %420, %arg208 : tensor<64x8x256xf32>
      %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %423 = stablehlo.subtract %419, %422 : tensor<64x8x256x256xf32>
      %424 = stablehlo.exponential %423 : tensor<64x8x256x256xf32>
      mhlo.return %424 : tensor<64x8x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256xf32>, tensor<512x256x256xf32>) -> tensor<64x8x256x256xf32>
    %375 = "mhlo.fusion"(%374, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x8x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [3] : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
      mhlo.return %417 : tensor<64x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x256xf32>, tensor<f32>) -> tensor<64x8x256xf32>
    %376 = "mhlo.fusion"(%374, %375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x8x256x256xf32>, %arg209: tensor<64x8x256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [0, 1, 2] : (tensor<64x8x256xf32>) -> tensor<64x8x256x256xf32>
      %418 = stablehlo.divide %arg208, %417 : tensor<64x8x256x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x256xf32>) -> tensor<512x256x256xf32>
      mhlo.return %419 : tensor<512x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8x256x256xf32>, tensor<64x8x256xf32>) -> tensor<512x256x256xf32>
    %377 = stablehlo.dot_general %365, %arg172, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %378 = "mhlo.fusion"(%377, %arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x256xf32>, %arg209: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x8x32xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,8,256,32]{3,1,2,0}"} : (tensor<64x256x8x32xf32>) -> tensor<64x8x256x32xf32>
      %421 = mhlo.copy %420 : tensor<64x8x256x32xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x8x256x32xf32>) -> tensor<512x256x32xf32>
      mhlo.return %422 : tensor<512x256x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x256xf32>, tensor<256xf32>) -> tensor<512x256x32xf32>
    %379 = stablehlo.dot_general %376, %378, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x256x256xf32>, tensor<512x256x32xf32>) -> tensor<512x256x32xf32>
    %380 = "mhlo.fusion"(%379) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<512x256x32xf32>):
      %417 = mhlo.bitcast %arg208 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x256x32xf32>) -> tensor<64x8x256x32xf32>
      %418 = stablehlo.transpose %417, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,8,32]{3,1,2,0}"} : (tensor<64x8x256x32xf32>) -> tensor<64x256x8x32xf32>
      %419 = mhlo.copy %418 : tensor<64x256x8x32xf32>
      %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x8x32xf32>) -> tensor<16384x256xf32>
      mhlo.return %420 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x32xf32>) -> tensor<16384x256xf32>
    %381 = stablehlo.dot_general %380, %arg174, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<256x256xf32>) -> tensor<16384x256xf32>
    %382 = "mhlo.fusion"(%359, %381, %arg175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<16384x256xf32>, %arg210: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x256x256xf32>
      mhlo.return %420 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %383 = "mhlo.fusion"(%382, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %384 = "mhlo.fusion"(%382, %383) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x256x256xf32>
      mhlo.return %421 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %385 = "mhlo.fusion"(%384, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
      mhlo.return %417 : tensor<64x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    %386 = "mhlo.fusion"(%385, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    %387 = "mhlo.fusion"(%386) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %388 = "mhlo.fusion"(%arg177, %arg176, %387, %382, %383) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256x256xf32>, %arg212: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x256x256xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x256x256xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x256x256xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<16384x256xf32>
      mhlo.return %427 : tensor<16384x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<16384x256xf32>
    %389 = stablehlo.dot_general %388, %arg178, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x256xf32>, tensor<1024x256xf32>) -> tensor<16384x1024xf32>
    %390 = "mhlo.fusion"(%389, %arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<16384x1024xf32>, %arg209: tensor<1024xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [1] : (tensor<1024xf32>) -> tensor<16384x1024xf32>
      %418 = stablehlo.add %417, %arg208 : tensor<16384x1024xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x1024xf32>) -> tensor<64x256x1024xf32>
      %420 = stablehlo.transpose %419, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,1024,256]{1,2,0}"} : (tensor<64x256x1024xf32>) -> tensor<64x1024x256xf32>
      %421 = mhlo.copy %420 : tensor<64x1024x256xf32>
      %422 = mhlo.bitcast %421 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x1024x256xf32>) -> tensor<64x1024x16x16xf32>
      %423 = stablehlo.transpose %422, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,16,16,1024]{2,1,3,0}"} : (tensor<64x1024x16x16xf32>) -> tensor<64x16x16x1024xf32>
      %424 = mhlo.copy %423 : tensor<64x16x16x1024xf32>
      mhlo.return %424 : tensor<64x16x16x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<16384x1024xf32>, tensor<1024xf32>) -> tensor<64x16x16x1024xf32>
    %391 = "mhlo.fusion"(%arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<1024x1x3x3xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,1024]{1,0,2,3}"} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
      %418 = mhlo.copy %417 : tensor<3x3x1x1024xf32>
      mhlo.return %418 : tensor<3x3x1x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1x3x3xf32>) -> tensor<3x3x1x1024xf32>
    %392 = stablehlo.convolution(%390, %391) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1024 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<64x16x16x1024xf32>, tensor<3x3x1x1024xf32>) -> tensor<64x16x16x1024xf32>
    %393 = "mhlo.fusion"(%392, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x16x16x1024xf32>, %arg209: tensor<1024xf32>):
      %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
      %418 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<1024xf32>) -> tensor<64x16x16x1024xf32>
      %419 = stablehlo.add %arg208, %418 : tensor<64x16x16x1024xf32>
      %420 = stablehlo.multiply %417, %419 : tensor<64x16x16x1024xf32>
      %421 = stablehlo.transpose %420, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
      %422 = mhlo.copy %421 : tensor<64x1024x16x16xf32>
      %423 = mhlo.bitcast %422 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %424 = stablehlo.transpose %423, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %425 = mhlo.copy %424 : tensor<64x256x1024xf32>
      %426 = stablehlo.negate %419 : tensor<64x16x16x1024xf32>
      %cst_3 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %427 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x16x16x1024xf32>
      %428 = stablehlo.multiply %426, %427 : tensor<64x16x16x1024xf32>
      %429 = stablehlo.transpose %428, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,1024,16,16]{1,3,2,0}"} : (tensor<64x16x16x1024xf32>) -> tensor<64x1024x16x16xf32>
      %430 = mhlo.copy %429 : tensor<64x1024x16x16xf32>
      %431 = mhlo.bitcast %430 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1024x16x16xf32>) -> tensor<64x1024x256xf32>
      %432 = stablehlo.transpose %431, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %433 = mhlo.copy %432 : tensor<64x256x1024xf32>
      %434 = stablehlo.abs %433 : tensor<64x256x1024xf32>
      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %435 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %437 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %438 = stablehlo.multiply %433, %433 : tensor<64x256x1024xf32>
      %439 = stablehlo.multiply %437, %438 : tensor<64x256x1024xf32>
      %cst_6 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %440 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %441 = stablehlo.add %439, %440 : tensor<64x256x1024xf32>
      %442 = stablehlo.multiply %441, %438 : tensor<64x256x1024xf32>
      %cst_7 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %443 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %444 = stablehlo.add %442, %443 : tensor<64x256x1024xf32>
      %445 = stablehlo.multiply %444, %438 : tensor<64x256x1024xf32>
      %cst_8 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %446 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %447 = stablehlo.add %445, %446 : tensor<64x256x1024xf32>
      %448 = stablehlo.multiply %447, %438 : tensor<64x256x1024xf32>
      %cst_9 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %449 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %450 = stablehlo.add %448, %449 : tensor<64x256x1024xf32>
      %451 = stablehlo.multiply %450, %438 : tensor<64x256x1024xf32>
      %cst_10 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %452 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %453 = stablehlo.add %451, %452 : tensor<64x256x1024xf32>
      %454 = stablehlo.multiply %453, %438 : tensor<64x256x1024xf32>
      %cst_11 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %455 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %456 = stablehlo.add %454, %455 : tensor<64x256x1024xf32>
      %457 = stablehlo.multiply %433, %456 : tensor<64x256x1024xf32>
      %458 = stablehlo.subtract %435, %457 : tensor<64x256x1024xf32>
      %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %459 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %460 = stablehlo.compare LT, %433, %459 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %cst_13 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %461 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %462 = stablehlo.negate %438 : tensor<64x256x1024xf32>
      %cst_14 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %464 = stablehlo.compare LT, %462, %463 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %465 = stablehlo.exponential %462 : tensor<64x256x1024xf32>
      %466 = stablehlo.divide %435, %434 : tensor<64x256x1024xf32>
      %467 = stablehlo.multiply %465, %466 : tensor<64x256x1024xf32>
      %468 = stablehlo.compare LT, %434, %461 : (tensor<64x256x1024xf32>, tensor<64x256x1024xf32>) -> tensor<64x256x1024xi1>
      %cst_15 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %469 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %470 = stablehlo.divide %435, %438 : tensor<64x256x1024xf32>
      %471 = stablehlo.multiply %469, %470 : tensor<64x256x1024xf32>
      %cst_16 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %472 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %473 = stablehlo.add %471, %472 : tensor<64x256x1024xf32>
      %474 = stablehlo.multiply %473, %470 : tensor<64x256x1024xf32>
      %cst_17 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %475 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %476 = stablehlo.add %474, %475 : tensor<64x256x1024xf32>
      %477 = stablehlo.multiply %476, %470 : tensor<64x256x1024xf32>
      %cst_18 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %478 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %479 = stablehlo.add %477, %478 : tensor<64x256x1024xf32>
      %480 = stablehlo.multiply %479, %470 : tensor<64x256x1024xf32>
      %cst_19 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %481 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %482 = stablehlo.add %480, %481 : tensor<64x256x1024xf32>
      %483 = stablehlo.multiply %482, %470 : tensor<64x256x1024xf32>
      %cst_20 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %484 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %485 = stablehlo.add %483, %484 : tensor<64x256x1024xf32>
      %486 = stablehlo.multiply %485, %470 : tensor<64x256x1024xf32>
      %cst_21 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %487 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %488 = stablehlo.add %486, %487 : tensor<64x256x1024xf32>
      %489 = stablehlo.multiply %488, %470 : tensor<64x256x1024xf32>
      %cst_22 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %491 = stablehlo.add %489, %490 : tensor<64x256x1024xf32>
      %492 = stablehlo.multiply %491, %470 : tensor<64x256x1024xf32>
      %cst_23 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %493 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %494 = stablehlo.add %492, %493 : tensor<64x256x1024xf32>
      %cst_24 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %495 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %496 = stablehlo.multiply %495, %470 : tensor<64x256x1024xf32>
      %cst_25 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %497 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %498 = stablehlo.add %496, %497 : tensor<64x256x1024xf32>
      %499 = stablehlo.multiply %498, %470 : tensor<64x256x1024xf32>
      %cst_26 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %500 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %501 = stablehlo.add %499, %500 : tensor<64x256x1024xf32>
      %502 = stablehlo.multiply %501, %470 : tensor<64x256x1024xf32>
      %cst_27 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %504 = stablehlo.add %502, %503 : tensor<64x256x1024xf32>
      %505 = stablehlo.multiply %504, %470 : tensor<64x256x1024xf32>
      %cst_28 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %506 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %507 = stablehlo.add %505, %506 : tensor<64x256x1024xf32>
      %508 = stablehlo.multiply %507, %470 : tensor<64x256x1024xf32>
      %cst_29 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %509 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %510 = stablehlo.add %508, %509 : tensor<64x256x1024xf32>
      %511 = stablehlo.multiply %510, %470 : tensor<64x256x1024xf32>
      %cst_30 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %512 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %513 = stablehlo.add %511, %512 : tensor<64x256x1024xf32>
      %514 = stablehlo.multiply %513, %470 : tensor<64x256x1024xf32>
      %cst_31 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %515 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<64x256x1024xf32>
      %516 = stablehlo.add %514, %515 : tensor<64x256x1024xf32>
      %517 = stablehlo.select %468, %494, %516 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %518 = stablehlo.multiply %467, %517 : tensor<64x256x1024xf32>
      %519 = stablehlo.select %464, %459, %518 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %520 = stablehlo.subtract %461, %519 : tensor<64x256x1024xf32>
      %521 = stablehlo.select %460, %520, %519 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %522 = stablehlo.select %436, %458, %521 : tensor<64x256x1024xi1>, tensor<64x256x1024xf32>
      %523 = stablehlo.multiply %425, %522 : tensor<64x256x1024xf32>
      %524 = mhlo.bitcast %523 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<16384x1024xf32>
      mhlo.return %524 : tensor<16384x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16x16x1024xf32>, tensor<1024xf32>) -> tensor<16384x1024xf32>
    %394 = stablehlo.dot_general %393, %arg182, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16384x1024xf32>, tensor<256x1024xf32>) -> tensor<16384x256xf32>
    %395 = "mhlo.fusion"(%382, %394, %arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<16384x256xf32>, %arg210: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg210, dims = [1] : (tensor<256xf32>) -> tensor<16384x256xf32>
      %418 = stablehlo.add %417, %arg209 : tensor<16384x256xf32>
      %419 = mhlo.bitcast %418 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<16384x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.add %419, %arg208 : tensor<64x256x256xf32>
      mhlo.return %420 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<16384x256xf32>, tensor<256xf32>) -> tensor<64x256x256xf32>
    %396 = "mhlo.fusion"(%395, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256xf32>
    %397 = "mhlo.fusion"(%395, %396) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg209, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg208, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.multiply %420, %420 : tensor<64x256x256xf32>
      mhlo.return %421 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %398 = "mhlo.fusion"(%397, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256x256xf32>, %arg209: tensor<f32>):
      %417 = "stablehlo.reduce_window"(%arg208, %arg209) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg210: tensor<f32>, %arg211: tensor<f32>):
        %418 = stablehlo.add %arg210, %arg211 : tensor<f32>
        stablehlo.return %418 : tensor<f32>
      }) : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
      mhlo.return %417 : tensor<64x256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x256xf32>, tensor<f32>) -> tensor<64x256x8xf32>
    %399 = "mhlo.fusion"(%398, %cst_0) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg208: tensor<64x256x8xf32>, %arg209: tensor<f32>):
      %417 = stablehlo.reduce(%arg208 init: %arg209) applies stablehlo.add across dimensions = [2] : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %417 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8xf32>, tensor<f32>) -> tensor<64x256xf32>
    %400 = "mhlo.fusion"(%399) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg208, %417 : tensor<64x256xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %420 = stablehlo.add %418, %419 : tensor<64x256xf32>
      %421 = stablehlo.rsqrt %420 : tensor<64x256xf32>
      mhlo.return %421 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64x256xf32>
    %401 = "mhlo.fusion"(%arg191, %arg190, %400, %395, %396) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<64x256xf32>, %arg211: tensor<64x256x256xf32>, %arg212: tensor<64x256xf32>):
      %cst_2 = stablehlo.constant dense<3.906250e-03> : tensor<f32>
      %417 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
      %418 = stablehlo.multiply %arg212, %417 : tensor<64x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %420 = stablehlo.subtract %arg211, %419 : tensor<64x256x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg210, dims = [0, 1] : (tensor<64x256xf32>) -> tensor<64x256x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x256x256xf32>
      %423 = stablehlo.broadcast_in_dim %arg209, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %424 = stablehlo.multiply %422, %423 : tensor<64x256x256xf32>
      %425 = stablehlo.broadcast_in_dim %arg208, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %426 = stablehlo.add %424, %425 : tensor<64x256x256xf32>
      %427 = mhlo.bitcast %426 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<64x16x16x256xf32>
      %428 = stablehlo.transpose %427, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,256,16,16]{1,3,2,0}"} : (tensor<64x16x16x256xf32>) -> tensor<64x256x16x16xf32>
      %429 = mhlo.copy %428 : tensor<64x256x16x16xf32>
      %430 = mhlo.bitcast %429 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x16x16xf32>) -> tensor<64x256x256xf32>
      mhlo.return %430 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256xf32>, tensor<64x256x256xf32>, tensor<64x256xf32>) -> tensor<64x256x256xf32>
    %402 = "mhlo.fusion"(%arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x256xf32>):
      %417 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[256,256]{0,1}"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %418 = mhlo.copy %417 : tensor<256x256xf32>
      %419 = stablehlo.broadcast_in_dim %418, dims = [1, 2] : (tensor<256x256xf32>) -> tensor<64x256x256xf32>
      mhlo.return %419 : tensor<64x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256xf32>) -> tensor<64x256x256xf32>
    %403 = stablehlo.dot_general %401, %402, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x256x256xf32>, tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
    %404 = "mhlo.fusion"(%1, %2, %86, %arg193, %87, %88, %89, %199, %arg195, %200, %201, %311, %arg197, %cst_0, %312, %313, %c, %cst_1, %403, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<128x128x2xi32>, %arg209: tensor<128x128x2xi32>, %arg210: tensor<64x16384x256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128x128x2xi32>, %arg213: tensor<128x128x2xi32>, %arg214: tensor<128xf32>, %arg215: tensor<64x4096x256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<128x128x2xi32>, %arg218: tensor<128x128x2xi32>, %arg219: tensor<64x1024x256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<f32>, %arg222: tensor<128x128x2xi32>, %arg223: tensor<128x128x2xi32>, %arg224: tensor<i32>, %arg225: tensor<f32>, %arg226: tensor<64x256x256xf32>, %arg227: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg227, dims = [2] : (tensor<256xf32>) -> tensor<64x256x256xf32>
      %418 = stablehlo.add %arg226, %417 : tensor<64x256x256xf32>
      %419 = stablehlo.transpose %418, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,256]{1,2,0}"} : (tensor<64x256x256xf32>) -> tensor<64x256x256xf32>
      %420 = mhlo.copy %419 : tensor<64x256x256xf32>
      %421 = mhlo.bitcast %420 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x256xf32>) -> tensor<64x256x16x16xf32>
      %422 = mhlo.bitcast %arg223 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %423 = "stablehlo.gather"(%421, %422) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %425 = stablehlo.transpose %424, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %426 = mhlo.copy %425 : tensor<64x256x128x128xf32>
      %427 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %cst_2 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %429 = stablehlo.multiply %arg214, %428 : tensor<128xf32>
      %430 = stablehlo.broadcast_in_dim %arg225, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %431 = stablehlo.add %429, %430 : tensor<128xf32>
      %432 = stablehlo.maximum %427, %431 : tensor<128xf32>
      %433 = mhlo.bitcast %432 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %434 = stablehlo.convert %433 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %435 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %436 = stablehlo.compare LT, %434, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_3 = stablehlo.constant dense<16> : tensor<i32>
      %437 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %438 = stablehlo.add %434, %437 : tensor<128x1xi32>
      %439 = stablehlo.select %436, %438, %434 : tensor<128x1xi1>, tensor<128x1xi32>
      %440 = mhlo.bitcast %439 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %441 = stablehlo.broadcast_in_dim %440, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_4 = stablehlo.constant dense<15> : tensor<i32>
      %442 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %443 = stablehlo.convert %432 : (tensor<128xf32>) -> tensor<128xi32>
      %c_5 = stablehlo.constant dense<1> : tensor<i32>
      %444 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %445 = stablehlo.add %443, %444 : tensor<128xi32>
      %446 = stablehlo.minimum %442, %445 : tensor<128xi32>
      %447 = stablehlo.broadcast_in_dim %arg224, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %448 = stablehlo.compare LT, %446, %447 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %449 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %450 = stablehlo.add %446, %449 : tensor<128xi32>
      %451 = stablehlo.select %448, %450, %446 : tensor<128xi1>, tensor<128xi32>
      %452 = stablehlo.broadcast_in_dim %451, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %453 = stablehlo.concatenate %441, %452, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %454 = mhlo.bitcast %453 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %455 = "stablehlo.gather"(%421, %454) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %456 = mhlo.bitcast %455 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %457 = stablehlo.transpose %456, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %458 = mhlo.copy %457 : tensor<64x256x128x128xf32>
      %459 = stablehlo.subtract %458, %426 : tensor<64x256x128x128xf32>
      %460 = stablehlo.convert %443 : (tensor<128xi32>) -> tensor<128xf32>
      %461 = stablehlo.subtract %432, %460 : tensor<128xf32>
      %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %462 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %463 = stablehlo.clamp %427, %461, %462 : tensor<128xf32>
      %464 = stablehlo.broadcast_in_dim %463, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %465 = stablehlo.multiply %459, %464 : tensor<64x256x128x128xf32>
      %466 = stablehlo.add %426, %465 : tensor<64x256x128x128xf32>
      %467 = mhlo.bitcast %arg222 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %468 = "stablehlo.gather"(%421, %467) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %469 = mhlo.bitcast %468 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %470 = stablehlo.transpose %469, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %471 = mhlo.copy %470 : tensor<64x256x128x128xf32>
      %472 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %473 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %474 = stablehlo.add %434, %473 : tensor<128x1xi32>
      %475 = stablehlo.minimum %472, %474 : tensor<128x1xi32>
      %476 = stablehlo.compare LT, %475, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %477 = stablehlo.add %475, %437 : tensor<128x1xi32>
      %478 = stablehlo.select %476, %477, %475 : tensor<128x1xi1>, tensor<128x1xi32>
      %479 = mhlo.bitcast %478 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %480 = stablehlo.broadcast_in_dim %479, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %481 = stablehlo.concatenate %480, %452, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %482 = mhlo.bitcast %481 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %483 = "stablehlo.gather"(%421, %482) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x16x16xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %484 = mhlo.bitcast %483 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %485 = stablehlo.transpose %484, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %486 = mhlo.copy %485 : tensor<64x256x128x128xf32>
      %487 = stablehlo.subtract %486, %471 : tensor<64x256x128x128xf32>
      %488 = stablehlo.multiply %487, %464 : tensor<64x256x128x128xf32>
      %489 = stablehlo.add %471, %488 : tensor<64x256x128x128xf32>
      %490 = stablehlo.subtract %489, %466 : tensor<64x256x128x128xf32>
      %491 = stablehlo.broadcast_in_dim %arg221, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %492 = stablehlo.convert %434 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %493 = stablehlo.subtract %433, %492 : tensor<128x1xf32>
      %494 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
      %495 = stablehlo.clamp %491, %493, %494 : tensor<128x1xf32>
      %496 = mhlo.bitcast %495 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %497 = stablehlo.broadcast_in_dim %496, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %498 = stablehlo.multiply %490, %497 : tensor<64x256x128x128xf32>
      %499 = stablehlo.add %466, %498 : tensor<64x256x128x128xf32>
      %500 = stablehlo.broadcast_in_dim %arg220, dims = [2] : (tensor<256xf32>) -> tensor<64x1024x256xf32>
      %501 = stablehlo.add %arg219, %500 : tensor<64x1024x256xf32>
      %502 = stablehlo.transpose %501, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,1024]{1,2,0}"} : (tensor<64x1024x256xf32>) -> tensor<64x256x1024xf32>
      %503 = mhlo.copy %502 : tensor<64x256x1024xf32>
      %504 = mhlo.bitcast %503 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x1024xf32>) -> tensor<64x256x32x32xf32>
      %505 = mhlo.bitcast %arg218 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %506 = "stablehlo.gather"(%504, %505) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %507 = mhlo.bitcast %506 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %508 = stablehlo.transpose %507, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %509 = mhlo.copy %508 : tensor<64x256x128x128xf32>
      %cst_7 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
      %510 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %511 = stablehlo.multiply %arg214, %510 : tensor<128xf32>
      %512 = stablehlo.add %511, %430 : tensor<128xf32>
      %513 = stablehlo.maximum %427, %512 : tensor<128xf32>
      %514 = mhlo.bitcast %513 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %515 = stablehlo.convert %514 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %516 = stablehlo.compare LT, %515, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_8 = stablehlo.constant dense<32> : tensor<i32>
      %517 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %518 = stablehlo.add %515, %517 : tensor<128x1xi32>
      %519 = stablehlo.select %516, %518, %515 : tensor<128x1xi1>, tensor<128x1xi32>
      %520 = mhlo.bitcast %519 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %521 = stablehlo.broadcast_in_dim %520, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_9 = stablehlo.constant dense<31> : tensor<i32>
      %522 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %523 = stablehlo.convert %513 : (tensor<128xf32>) -> tensor<128xi32>
      %524 = stablehlo.add %523, %444 : tensor<128xi32>
      %525 = stablehlo.minimum %522, %524 : tensor<128xi32>
      %526 = stablehlo.compare LT, %525, %447 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %527 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %528 = stablehlo.add %525, %527 : tensor<128xi32>
      %529 = stablehlo.select %526, %528, %525 : tensor<128xi1>, tensor<128xi32>
      %530 = stablehlo.broadcast_in_dim %529, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %531 = stablehlo.concatenate %521, %530, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %532 = mhlo.bitcast %531 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %533 = "stablehlo.gather"(%504, %532) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %534 = mhlo.bitcast %533 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %535 = stablehlo.transpose %534, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %536 = mhlo.copy %535 : tensor<64x256x128x128xf32>
      %537 = stablehlo.subtract %536, %509 : tensor<64x256x128x128xf32>
      %538 = stablehlo.convert %523 : (tensor<128xi32>) -> tensor<128xf32>
      %539 = stablehlo.subtract %513, %538 : tensor<128xf32>
      %540 = stablehlo.clamp %427, %539, %462 : tensor<128xf32>
      %541 = stablehlo.broadcast_in_dim %540, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %542 = stablehlo.multiply %537, %541 : tensor<64x256x128x128xf32>
      %543 = stablehlo.add %509, %542 : tensor<64x256x128x128xf32>
      %544 = mhlo.bitcast %arg217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %545 = "stablehlo.gather"(%504, %544) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %546 = mhlo.bitcast %545 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %547 = stablehlo.transpose %546, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %548 = mhlo.copy %547 : tensor<64x256x128x128xf32>
      %549 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %550 = stablehlo.add %515, %473 : tensor<128x1xi32>
      %551 = stablehlo.minimum %549, %550 : tensor<128x1xi32>
      %552 = stablehlo.compare LT, %551, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %553 = stablehlo.add %551, %517 : tensor<128x1xi32>
      %554 = stablehlo.select %552, %553, %551 : tensor<128x1xi1>, tensor<128x1xi32>
      %555 = mhlo.bitcast %554 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %556 = stablehlo.broadcast_in_dim %555, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %557 = stablehlo.concatenate %556, %530, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %558 = mhlo.bitcast %557 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %559 = "stablehlo.gather"(%504, %558) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x32x32xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %560 = mhlo.bitcast %559 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %561 = stablehlo.transpose %560, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %562 = mhlo.copy %561 : tensor<64x256x128x128xf32>
      %563 = stablehlo.subtract %562, %548 : tensor<64x256x128x128xf32>
      %564 = stablehlo.multiply %563, %541 : tensor<64x256x128x128xf32>
      %565 = stablehlo.add %548, %564 : tensor<64x256x128x128xf32>
      %566 = stablehlo.subtract %565, %543 : tensor<64x256x128x128xf32>
      %567 = stablehlo.convert %515 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %568 = stablehlo.subtract %514, %567 : tensor<128x1xf32>
      %569 = stablehlo.clamp %491, %568, %494 : tensor<128x1xf32>
      %570 = mhlo.bitcast %569 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %571 = stablehlo.broadcast_in_dim %570, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %572 = stablehlo.multiply %566, %571 : tensor<64x256x128x128xf32>
      %573 = stablehlo.add %543, %572 : tensor<64x256x128x128xf32>
      %574 = stablehlo.broadcast_in_dim %arg216, dims = [2] : (tensor<256xf32>) -> tensor<64x4096x256xf32>
      %575 = stablehlo.add %arg215, %574 : tensor<64x4096x256xf32>
      %576 = stablehlo.transpose %575, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,4096]{1,2,0}"} : (tensor<64x4096x256xf32>) -> tensor<64x256x4096xf32>
      %577 = mhlo.copy %576 : tensor<64x256x4096xf32>
      %578 = mhlo.bitcast %577 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x4096xf32>) -> tensor<64x256x64x64xf32>
      %579 = mhlo.bitcast %arg213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %580 = "stablehlo.gather"(%578, %579) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %581 = mhlo.bitcast %580 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %582 = stablehlo.transpose %581, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %583 = mhlo.copy %582 : tensor<64x256x128x128xf32>
      %cst_10 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %584 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %585 = stablehlo.multiply %arg214, %584 : tensor<128xf32>
      %586 = stablehlo.add %585, %430 : tensor<128xf32>
      %587 = stablehlo.maximum %427, %586 : tensor<128xf32>
      %588 = mhlo.bitcast %587 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %589 = stablehlo.convert %588 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %590 = stablehlo.compare LT, %589, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_11 = stablehlo.constant dense<64> : tensor<i32>
      %591 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %592 = stablehlo.add %589, %591 : tensor<128x1xi32>
      %593 = stablehlo.select %590, %592, %589 : tensor<128x1xi1>, tensor<128x1xi32>
      %594 = mhlo.bitcast %593 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %595 = stablehlo.broadcast_in_dim %594, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_12 = stablehlo.constant dense<63> : tensor<i32>
      %596 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %597 = stablehlo.convert %587 : (tensor<128xf32>) -> tensor<128xi32>
      %598 = stablehlo.add %597, %444 : tensor<128xi32>
      %599 = stablehlo.minimum %596, %598 : tensor<128xi32>
      %600 = stablehlo.compare LT, %599, %447 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %601 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %602 = stablehlo.add %599, %601 : tensor<128xi32>
      %603 = stablehlo.select %600, %602, %599 : tensor<128xi1>, tensor<128xi32>
      %604 = stablehlo.broadcast_in_dim %603, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %605 = stablehlo.concatenate %595, %604, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %606 = mhlo.bitcast %605 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %607 = "stablehlo.gather"(%578, %606) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %608 = mhlo.bitcast %607 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %609 = stablehlo.transpose %608, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %610 = mhlo.copy %609 : tensor<64x256x128x128xf32>
      %611 = stablehlo.subtract %610, %583 : tensor<64x256x128x128xf32>
      %612 = stablehlo.convert %597 : (tensor<128xi32>) -> tensor<128xf32>
      %613 = stablehlo.subtract %587, %612 : tensor<128xf32>
      %614 = stablehlo.clamp %427, %613, %462 : tensor<128xf32>
      %615 = stablehlo.broadcast_in_dim %614, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %616 = stablehlo.multiply %611, %615 : tensor<64x256x128x128xf32>
      %617 = stablehlo.add %583, %616 : tensor<64x256x128x128xf32>
      %618 = mhlo.bitcast %arg212 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %619 = "stablehlo.gather"(%578, %618) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %620 = mhlo.bitcast %619 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %621 = stablehlo.transpose %620, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %622 = mhlo.copy %621 : tensor<64x256x128x128xf32>
      %623 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %624 = stablehlo.add %589, %473 : tensor<128x1xi32>
      %625 = stablehlo.minimum %623, %624 : tensor<128x1xi32>
      %626 = stablehlo.compare LT, %625, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %627 = stablehlo.add %625, %591 : tensor<128x1xi32>
      %628 = stablehlo.select %626, %627, %625 : tensor<128x1xi1>, tensor<128x1xi32>
      %629 = mhlo.bitcast %628 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %630 = stablehlo.broadcast_in_dim %629, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %631 = stablehlo.concatenate %630, %604, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %632 = mhlo.bitcast %631 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %633 = "stablehlo.gather"(%578, %632) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x64x64xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %634 = mhlo.bitcast %633 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %635 = stablehlo.transpose %634, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %636 = mhlo.copy %635 : tensor<64x256x128x128xf32>
      %637 = stablehlo.subtract %636, %622 : tensor<64x256x128x128xf32>
      %638 = stablehlo.multiply %637, %615 : tensor<64x256x128x128xf32>
      %639 = stablehlo.add %622, %638 : tensor<64x256x128x128xf32>
      %640 = stablehlo.subtract %639, %617 : tensor<64x256x128x128xf32>
      %641 = stablehlo.convert %589 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %642 = stablehlo.subtract %588, %641 : tensor<128x1xf32>
      %643 = stablehlo.clamp %491, %642, %494 : tensor<128x1xf32>
      %644 = mhlo.bitcast %643 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %646 = stablehlo.multiply %640, %645 : tensor<64x256x128x128xf32>
      %647 = stablehlo.add %617, %646 : tensor<64x256x128x128xf32>
      %648 = stablehlo.broadcast_in_dim %arg211, dims = [2] : (tensor<256xf32>) -> tensor<64x16384x256xf32>
      %649 = stablehlo.add %arg210, %648 : tensor<64x16384x256xf32>
      %650 = stablehlo.transpose %649, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[64,256,16384]{1,2,0}"} : (tensor<64x16384x256xf32>) -> tensor<64x256x16384xf32>
      %651 = mhlo.copy %650 : tensor<64x256x16384xf32>
      %652 = mhlo.bitcast %651 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<64x256x16384xf32>) -> tensor<64x256x128x128xf32>
      %653 = mhlo.bitcast %arg209 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %654 = "stablehlo.gather"(%652, %653) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %656 = stablehlo.transpose %655, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %657 = mhlo.copy %656 : tensor<64x256x128x128xf32>
      %658 = stablehlo.iota dim = 0 : tensor<128xi32>
      %659 = stablehlo.convert %658 : (tensor<128xi32>) -> tensor<128xf32>
      %660 = stablehlo.maximum %427, %659 : tensor<128xf32>
      %661 = mhlo.bitcast %660 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<0> : tensor<1xindex>} : (tensor<128xf32>) -> tensor<128x1xf32>
      %662 = stablehlo.convert %661 : (tensor<128x1xf32>) -> tensor<128x1xi32>
      %663 = stablehlo.compare LT, %662, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %c_13 = stablehlo.constant dense<128> : tensor<i32>
      %664 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %665 = stablehlo.add %662, %664 : tensor<128x1xi32>
      %666 = stablehlo.select %663, %665, %662 : tensor<128x1xi1>, tensor<128x1xi32>
      %667 = mhlo.bitcast %666 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %668 = stablehlo.broadcast_in_dim %667, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %c_14 = stablehlo.constant dense<127> : tensor<i32>
      %669 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %670 = stablehlo.convert %660 : (tensor<128xf32>) -> tensor<128xi32>
      %671 = stablehlo.add %670, %444 : tensor<128xi32>
      %672 = stablehlo.minimum %669, %671 : tensor<128xi32>
      %673 = stablehlo.compare LT, %672, %447 : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi1>
      %674 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<128xi32>
      %675 = stablehlo.add %672, %674 : tensor<128xi32>
      %676 = stablehlo.select %673, %675, %672 : tensor<128xi1>, tensor<128xi32>
      %677 = stablehlo.broadcast_in_dim %676, dims = [1] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %678 = stablehlo.concatenate %668, %677, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %679 = mhlo.bitcast %678 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %680 = "stablehlo.gather"(%652, %679) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %681 = mhlo.bitcast %680 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %682 = stablehlo.transpose %681, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %683 = mhlo.copy %682 : tensor<64x256x128x128xf32>
      %684 = stablehlo.subtract %683, %657 : tensor<64x256x128x128xf32>
      %685 = stablehlo.convert %670 : (tensor<128xi32>) -> tensor<128xf32>
      %686 = stablehlo.subtract %660, %685 : tensor<128xf32>
      %687 = stablehlo.clamp %427, %686, %462 : tensor<128xf32>
      %688 = stablehlo.broadcast_in_dim %687, dims = [3] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %689 = stablehlo.multiply %684, %688 : tensor<64x256x128x128xf32>
      %690 = stablehlo.add %657, %689 : tensor<64x256x128x128xf32>
      %691 = mhlo.bitcast %arg208 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %692 = "stablehlo.gather"(%652, %691) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %693 = mhlo.bitcast %692 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %694 = stablehlo.transpose %693, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %695 = mhlo.copy %694 : tensor<64x256x128x128xf32>
      %696 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<128x1xi32>
      %697 = stablehlo.add %662, %473 : tensor<128x1xi32>
      %698 = stablehlo.minimum %696, %697 : tensor<128x1xi32>
      %699 = stablehlo.compare LT, %698, %435 : (tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi1>
      %700 = stablehlo.add %698, %664 : tensor<128x1xi32>
      %701 = stablehlo.select %699, %700, %698 : tensor<128x1xi1>, tensor<128x1xi32>
      %702 = mhlo.bitcast %701 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xi32>) -> tensor<128xi32>
      %703 = stablehlo.broadcast_in_dim %702, dims = [0] : (tensor<128xi32>) -> tensor<128x128x1xi32>
      %704 = stablehlo.concatenate %703, %677, dim = 2 : (tensor<128x128x1xi32>, tensor<128x128x1xi32>) -> tensor<128x128x2xi32>
      %705 = mhlo.bitcast %704 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<128x128x2xi32>) -> tensor<16384x2xi32>
      %706 = "stablehlo.gather"(%652, %705) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x128x128xf32>, tensor<16384x2xi32>) -> tensor<16384x64x256x1x1xf32>
      %707 = mhlo.bitcast %706 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<16384x64x256x1x1xf32>) -> tensor<128x128x64x256xf32>
      %708 = stablehlo.transpose %707, dims = [2, 3, 0, 1] {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, xla_shape = "f32[64,256,128,128]{1,0,3,2}"} : (tensor<128x128x64x256xf32>) -> tensor<64x256x128x128xf32>
      %709 = mhlo.copy %708 : tensor<64x256x128x128xf32>
      %710 = stablehlo.subtract %709, %695 : tensor<64x256x128x128xf32>
      %711 = stablehlo.multiply %710, %688 : tensor<64x256x128x128xf32>
      %712 = stablehlo.add %695, %711 : tensor<64x256x128x128xf32>
      %713 = stablehlo.subtract %712, %690 : tensor<64x256x128x128xf32>
      %714 = stablehlo.convert %662 : (tensor<128x1xi32>) -> tensor<128x1xf32>
      %715 = stablehlo.subtract %661, %714 : tensor<128x1xf32>
      %716 = stablehlo.clamp %491, %715, %494 : tensor<128x1xf32>
      %717 = mhlo.bitcast %716 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<128x1xf32>) -> tensor<128xf32>
      %718 = stablehlo.broadcast_in_dim %717, dims = [2] : (tensor<128xf32>) -> tensor<64x256x128x128xf32>
      %719 = stablehlo.multiply %713, %718 : tensor<64x256x128x128xf32>
      %720 = stablehlo.add %690, %719 : tensor<64x256x128x128xf32>
      %721 = stablehlo.concatenate %499, %573, %647, %720, dim = 1 : (tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>, tensor<64x256x128x128xf32>) -> tensor<64x1024x128x128xf32>
      %722 = stablehlo.transpose %721, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[64,128,128,1024]{2,1,3,0}"} : (tensor<64x1024x128x128xf32>) -> tensor<64x128x128x1024xf32>
      %723 = mhlo.copy %722 : tensor<64x128x128x1024xf32>
      mhlo.return %723 : tensor<64x128x128x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<64x16384x256xf32>, tensor<256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<128xf32>, tensor<64x4096x256xf32>, tensor<256xf32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<64x1024x256xf32>, tensor<256xf32>, tensor<f32>, tensor<128x128x2xi32>, tensor<128x128x2xi32>, tensor<i32>, tensor<f32>, tensor<64x256x256xf32>, tensor<256xf32>) -> tensor<64x128x128x1024xf32>
    %405 = mhlo.bitcast %404 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x1024xf32>) -> tensor<1048576x1024xf32>
    %406 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256x1024x1x1xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,1024,256]{1,0,2,3}"} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
      %418 = mhlo.copy %417 : tensor<1x1x1024x256xf32>
      mhlo.return %418 : tensor<1x1x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024x1x1xf32>) -> tensor<1x1x1024x256xf32>
    %407 = mhlo.bitcast %406 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1024x256xf32>) -> tensor<1024x256xf32>
    %408 = stablehlo.dot %405, %407, precision = [DEFAULT, DEFAULT] : (tensor<1048576x1024xf32>, tensor<1024x256xf32>) -> tensor<1048576x256xf32>
    %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x256xf32>) -> tensor<64x128x128x256xf32>
    %410 = "mhlo.fusion"(%arg202, %arg201, %0, %409, %arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<64x128x128x256xf32>, %arg212: tensor<256xf32>):
      %417 = stablehlo.broadcast_in_dim %arg212, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
      %418 = stablehlo.subtract %arg211, %417 : tensor<64x128x128x256xf32>
      %419 = stablehlo.broadcast_in_dim %arg210, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
      %420 = stablehlo.multiply %418, %419 : tensor<64x128x128x256xf32>
      %421 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
      %422 = stablehlo.multiply %420, %421 : tensor<64x128x128x256xf32>
      %423 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<256xf32>) -> tensor<64x128x128x256xf32>
      %424 = stablehlo.add %422, %423 : tensor<64x128x128x256xf32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x128x128x256xf32>
      %426 = stablehlo.maximum %424, %425 : tensor<64x128x128x256xf32>
      mhlo.return %426 : tensor<64x128x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x128x128x256xf32>, tensor<256xf32>) -> tensor<64x128x128x256xf32>
    %411 = mhlo.bitcast %410 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x128x256xf32>) -> tensor<1048576x256xf32>
    %412 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<150x256x1x1xf32>):
      %417 = stablehlo.transpose %arg208, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,150]{1,0,2,3}"} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
      %418 = mhlo.copy %417 : tensor<1x1x256x150xf32>
      mhlo.return %418 : tensor<1x1x256x150xf32>
    }) {output_operand_aliasing = []} : (tensor<150x256x1x1xf32>) -> tensor<1x1x256x150xf32>
    %413 = mhlo.bitcast %412 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x150xf32>) -> tensor<256x150xf32>
    %414 = stablehlo.dot %411, %413, precision = [DEFAULT, DEFAULT] : (tensor<1048576x256xf32>, tensor<256x150xf32>) -> tensor<1048576x150xf32>
    %415 = mhlo.bitcast %414 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1048576x150xf32>) -> tensor<64x128x128x150xf32>
    %416 = "mhlo.fusion"(%415, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg208: tensor<64x128x128x150xf32>, %arg209: tensor<150xf32>):
      %417 = stablehlo.broadcast_in_dim %arg209, dims = [3] : (tensor<150xf32>) -> tensor<64x128x128x150xf32>
      %418 = stablehlo.add %arg208, %417 : tensor<64x128x128x150xf32>
      %419 = stablehlo.transpose %418, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[64,150,128,128]{1,3,2,0}"} : (tensor<64x128x128x150xf32>) -> tensor<64x150x128x128xf32>
      %420 = mhlo.copy %419 : tensor<64x150x128x128xf32>
      mhlo.return %420 : tensor<64x150x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x128x150xf32>, tensor<150xf32>) -> tensor<64x150x128x128xf32>
    return %416 : tensor<64x150x128x128xf32>
  }
}
