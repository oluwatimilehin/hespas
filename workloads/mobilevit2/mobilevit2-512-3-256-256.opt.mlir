module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_99.128(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<512x8x8x512xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x8x8x512xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_69.88.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_90.117(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_69.88.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_81.106(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_69.88.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_72.94(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_69.88.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<512x512xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_27.35.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_58.76.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<512x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    return %0 : tensor<512x192x4xf32>
  }
  func.func private @region_57.75(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_27.35.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_27.35.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_49.65.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<512x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    return %0 : tensor<512x192x4xf32>
  }
  func.func private @region_48.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_27.35.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_27.35.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_40.54.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<512x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    return %0 : tensor<512x192x4xf32>
  }
  func.func private @region_39.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_27.35.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.23(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_27.35.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.24(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_31.42.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.25(%arg0: tensor<512x192x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x192x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    return %0 : tensor<512x192x4xf32>
  }
  func.func private @region_30.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_27.35.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<512x1536xf32>, %arg1: tensor<f32>) -> tensor<512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    return %0 : tensor<512x2xf32>
  }
  func.func private @region_3.3.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.28(%arg0: tensor<512x4096xf32>, %arg1: tensor<f32>) -> tensor<512x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    return %0 : tensor<512x4xf32>
  }
  func.func private @region_21.28.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.29(%arg0: tensor<1x512x128xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_3.3.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<512x4096xf32>, %arg1: tensor<f32>) -> tensor<512x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    return %0 : tensor<512x4xf32>
  }
  func.func private @region_18.24.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.31(%arg0: tensor<1x512x128xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_16.23.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<512x128x4x8xf32>, %arg1: tensor<f32>) -> tensor<512x128x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x128x4x8xf32>, tensor<f32>) -> tensor<512x128x4xf32>
    return %0 : tensor<512x128x4xf32>
  }
  func.func private @region_15.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_3.3.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.34(%arg0: tensor<512x4096xf32>, %arg1: tensor<f32>) -> tensor<512x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    return %0 : tensor<512x4xf32>
  }
  func.func private @region_12.17.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<1x512x128xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_3.3.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.36(%arg0: tensor<512x4096xf32>, %arg1: tensor<f32>) -> tensor<512x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    return %0 : tensor<512x4xf32>
  }
  func.func private @region_9.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.37(%arg0: tensor<1x512x128xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_7.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.38(%arg0: tensor<512x128x4x8xf32>, %arg1: tensor<f32>) -> tensor<512x128x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x128x4x8xf32>, tensor<f32>) -> tensor<512x128x4xf32>
    return %0 : tensor<512x128x4xf32>
  }
  func.func private @region_6.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.39(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_3.3.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.40(%arg0: tensor<512x4096xf32>, %arg1: tensor<f32>) -> tensor<512x4xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    return %0 : tensor<512x4xf32>
  }
  func.func private @region_1.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.41(%arg0: tensor<1x512x128xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @fused_computation.42(%arg0: tensor<512x2xf32>, %arg1: tensor<2xf32>) -> tensor<512x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<512x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<512x2xf32>
    return %1 : tensor<512x2xf32>
  }
  func.func private @fused_computation.43(%arg0: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x512xf32>
    return %1 : tensor<512x512xf32>
  }
  func.func private @fused_computation.44(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x8x8x512xf32>, %arg4: tensor<512xf32>) -> tensor<512x8x8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x8x8x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x8x8x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x8x8x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x8x8x512xf32>
    return %7 : tensor<512x8x8x512xf32>
  }
  func.func private @fused_computation.45(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.46(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.47(%arg0: tensor<512x256x8x8xf32>) -> tensor<512x8x8x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,8,8,256]{2,1,3,0}"} : (tensor<512x256x8x8xf32>) -> tensor<512x8x8x256xf32>
    %1 = mhlo.copy %0 : tensor<512x8x8x256xf32>
    return %1 : tensor<512x8x8x256xf32>
  }
  func.func private @fused_computation.48(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256x4x16xf32>, %arg4: tensor<512x4x16x256xf32>, %arg5: tensor<256xf32>, %arg6: tensor<256xf32>) -> tensor<64x512x256x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %2 = stablehlo.add %arg4, %1 : tensor<512x4x16x256xf32>
    %3 = stablehlo.transpose %2, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %4 = mhlo.copy %3 : tensor<512x256x4x16xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<512x256x4x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<512xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<512x16384xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<512x16384xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<512x256x4x16xf32>
    %16 = stablehlo.add %0, %15 : tensor<512x256x4x16xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x256x2x2x4x4xf32>
    %18 = stablehlo.transpose %17, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,4,2,4,512,256]{3,1,2,0,5,4}"} : (tensor<512x256x2x2x4x4xf32>) -> tensor<2x4x2x4x512x256xf32>
    %19 = mhlo.copy %18 : tensor<2x4x2x4x512x256xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[64,512,256,1,1]{2,1,0,4,3}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<64x512x256x1x1xf32>
    %21 = mhlo.copy %20 : tensor<64x512x256x1x1xf32>
    return %21 : tensor<64x512x256x1x1xf32>
  }
  func.func private @fused_computation.49(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.50(%arg0: tensor<512xf32>, %arg1: tensor<512x256x4x16xf32>, %arg2: tensor<512x4x16x256xf32>, %arg3: tensor<256xf32>) -> tensor<512x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<512x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<512xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %9 = stablehlo.subtract %5, %8 : tensor<512x16384xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<512x16384xf32>
    return %10 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.51(%arg0: tensor<512x256x4x16xf32>, %arg1: tensor<512x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x512x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
    return %5 : tensor<1x512x16384xf32>
  }
  func.func private @fused_computation.52(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.53(%arg0: tensor<512x4x16x512xf32>, %arg1: tensor<512xf32>) -> tensor<512x4x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x4x16x512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x512xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,4,16]{1,3,2,0}"} : (tensor<512x4x16x512xf32>) -> tensor<512x512x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x512x4x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512x4x16xf32>
    %5 = stablehlo.negate %3 : tensor<512x512x4x16xf32>
    %6 = stablehlo.exponential %5 : tensor<512x512x4x16xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x512x4x16xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x512x4x16xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x512x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,512]{2,1,3,0}"} : (tensor<512x512x4x16xf32>) -> tensor<512x4x16x512xf32>
    %11 = mhlo.copy %10 : tensor<512x4x16x512xf32>
    return %11 : tensor<512x4x16x512xf32>
  }
  func.func private @fused_computation.54(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.55(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<512x4x16x256xf32>
    return %13 : tensor<512x4x16x256xf32>
  }
  func.func private @fused_computation.56(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.57(%arg0: tensor<512xf32>, %arg1: tensor<512x256x4x16xf32>) -> tensor<512x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x16384xf32>
    return %5 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.58(%arg0: tensor<512x256x4x16xf32>, %arg1: tensor<512x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<512x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x256x4x16xf32>
    return %4 : tensor<512x256x4x16xf32>
  }
  func.func private @fused_computation.59(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<512x256x4xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 257:513, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x4x16xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x256x4x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x256x4xf32>) -> tensor<512x256x4x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x256x4x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %10 = mhlo.copy %9 : tensor<512x4x16x256xf32>
    return %10 : tensor<512x4x16x256xf32>
  }
  func.func private @region_91.118(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.61(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x256x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 1:257, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x16xf32>) -> tensor<512x4x16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x16xf32>) -> tensor<512x256x4x16xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<512x256x4x16xf32>, tensor<f32>) -> tensor<512x256x4xf32>
    return %8 : tensor<512x256x4xf32>
  }
  func.func private @fused_computation.62(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x16xf32>
    return %2 : tensor<512x1x4x16xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x1x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x16xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x16xf32>
    return %10 : tensor<512x1x4x16xf32>
  }
  func.func private @region_89.116(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.64(%arg0: tensor<512x4x16x513xf32>, %arg1: tensor<513xf32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %5 : tensor<512x1x4xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x513xf32>
    return %1 : tensor<1x1x256x513xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<512x4x16x256xf32>
    return %13 : tensor<512x4x16x256xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<512xf32>, %arg1: tensor<512x256x4x16xf32>) -> tensor<512x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x16384xf32>
    return %5 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.69(%arg0: tensor<512x256x4x16xf32>, %arg1: tensor<512x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<512x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x256x4x16xf32>
    return %4 : tensor<512x256x4x16xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<512x4x16x512xf32>, %arg1: tensor<512xf32>) -> tensor<512x4x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x4x16x512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x512xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,4,16]{1,3,2,0}"} : (tensor<512x4x16x512xf32>) -> tensor<512x512x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x512x4x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512x4x16xf32>
    %5 = stablehlo.negate %3 : tensor<512x512x4x16xf32>
    %6 = stablehlo.exponential %5 : tensor<512x512x4x16xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x512x4x16xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x512x4x16xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x512x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,512]{2,1,3,0}"} : (tensor<512x512x4x16xf32>) -> tensor<512x4x16x512xf32>
    %11 = mhlo.copy %10 : tensor<512x4x16x512xf32>
    return %11 : tensor<512x4x16x512xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<512x4x16x256xf32>
    return %13 : tensor<512x4x16x256xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<512xf32>, %arg1: tensor<512x256x4x16xf32>) -> tensor<512x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x16384xf32>
    return %5 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<512x256x4x16xf32>, %arg1: tensor<512x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<512x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x256x4x16xf32>
    return %4 : tensor<512x256x4x16xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<512x256x4xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 257:513, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x4x16xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x256x4x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x256x4xf32>) -> tensor<512x256x4x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x256x4x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %10 = mhlo.copy %9 : tensor<512x4x16x256xf32>
    return %10 : tensor<512x4x16x256xf32>
  }
  func.func private @region_82.107(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.79(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x256x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 1:257, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x16xf32>) -> tensor<512x4x16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x16xf32>) -> tensor<512x256x4x16xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<512x256x4x16xf32>, tensor<f32>) -> tensor<512x256x4xf32>
    return %8 : tensor<512x256x4xf32>
  }
  func.func private @fused_computation.80(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x16xf32>
    return %2 : tensor<512x1x4x16xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x1x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x16xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x16xf32>
    return %10 : tensor<512x1x4x16xf32>
  }
  func.func private @region_80.105(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.82(%arg0: tensor<512x4x16x513xf32>, %arg1: tensor<513xf32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %5 : tensor<512x1x4xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x513xf32>
    return %1 : tensor<1x1x256x513xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<512x4x16x256xf32>
    return %13 : tensor<512x4x16x256xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<512xf32>, %arg1: tensor<512x256x4x16xf32>) -> tensor<512x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x16384xf32>
    return %5 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<512x256x4x16xf32>, %arg1: tensor<512x4x16x256xf32>, %arg2: tensor<256xf32>) -> tensor<512x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x256x4x16xf32>
    return %4 : tensor<512x256x4x16xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<512x4x16x512xf32>, %arg1: tensor<512xf32>) -> tensor<512x4x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x4x16x512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x512xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,4,16]{1,3,2,0}"} : (tensor<512x4x16x512xf32>) -> tensor<512x512x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x512x4x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512x4x16xf32>
    %5 = stablehlo.negate %3 : tensor<512x512x4x16xf32>
    %6 = stablehlo.exponential %5 : tensor<512x512x4x16xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x512x4x16xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x512x4x16xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x512x4x16xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,512]{2,1,3,0}"} : (tensor<512x512x4x16xf32>) -> tensor<512x4x16x512xf32>
    %11 = mhlo.copy %10 : tensor<512x4x16x512xf32>
    return %11 : tensor<512x4x16x512xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.91(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x256x4x16xf32>, %arg4: tensor<256xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x16384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x16384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x256x4x16xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x256x4x16xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %13 = mhlo.copy %12 : tensor<512x4x16x256xf32>
    return %13 : tensor<512x4x16x256xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<512xf32>, %arg1: tensor<512x256x4x16xf32>) -> tensor<512x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x16384xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x16384xf32>
    return %5 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<512x4x16x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x512x256x1x1xf32>) -> tensor<512x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x16xf32>
    %4 = mhlo.bitcast %arg2 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
    %5 = stablehlo.transpose %4, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
    %6 = mhlo.copy %5 : tensor<512x256x2x2x4x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<512x256x4x16xf32>
    %8 = stablehlo.add %3, %7 : tensor<512x256x4x16xf32>
    return %8 : tensor<512x256x4x16xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<512x256x4xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 257:513, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x4x16xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x256x4x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x256x4xf32>) -> tensor<512x256x4x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x256x4x16xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %10 = mhlo.copy %9 : tensor<512x4x16x256xf32>
    return %10 : tensor<512x4x16x256xf32>
  }
  func.func private @region_73.95(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.97(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x256x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 1:257, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x16xf32>) -> tensor<512x4x16xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x16xf32>) -> tensor<512x256x4x16xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<512x256x4x16xf32>, tensor<f32>) -> tensor<512x256x4xf32>
    return %8 : tensor<512x256x4xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<512x1x4x16xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x16xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x16xf32>
    return %2 : tensor<512x1x4x16xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x16x513xf32>, %arg2: tensor<513xf32>) -> tensor<512x1x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x16xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x16xf32>
    return %10 : tensor<512x1x4x16xf32>
  }
  func.func private @region_71.93(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.100(%arg0: tensor<512x4x16x513xf32>, %arg1: tensor<513xf32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x16x513xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
    %3 = mhlo.copy %2 : tensor<512x513x4x16xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.reduce(%4 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %5 : tensor<512x1x4xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x513xf32>
    return %1 : tensor<1x1x256x513xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<256xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<64x512x256x1x1xf32>, %arg4: tensor<256xf32>) -> tensor<512x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
    %2 = stablehlo.transpose %1, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
    %3 = mhlo.copy %2 : tensor<512x256x2x2x4x4xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6 = stablehlo.multiply %arg2, %5 : tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<512x16384xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x16384xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<512x256x4x16xf32>
    %14 = stablehlo.add %0, %13 : tensor<512x256x4x16xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
    %16 = mhlo.copy %15 : tensor<512x4x16x256xf32>
    return %16 : tensor<512x4x16x256xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %1, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<512xf32>, %arg1: tensor<64x512x256x1x1xf32>) -> tensor<512x16384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
    %2 = mhlo.copy %1 : tensor<512x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<512x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
    %7 = stablehlo.subtract %3, %6 : tensor<512x16384xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x16384xf32>
    return %8 : tensor<512x16384xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<64x512x256x1x1xf32>) -> tensor<1x512x16384xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
    %2 = mhlo.copy %1 : tensor<512x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<1x512x16384xf32>
    return %3 : tensor<1x512x16384xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<512x8x8x256xf32>, %arg1: tensor<2x4x2x4x2xi32>) -> tensor<64x512x256x1x1xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,8,8]{1,3,2,0}"} : (tensor<512x8x8x256xf32>) -> tensor<512x256x8x8xf32>
    %1 = mhlo.copy %0 : tensor<512x256x8x8xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    %3 = "stablehlo.gather"(%1, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 512, 256, 1, 1>}> : (tensor<512x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x512x256x1x1xf32>
    return %3 : tensor<64x512x256x1x1xf32>
  }
  func.func private @fused_computation.107(%arg0: tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x256xf32>
    return %1 : tensor<1x1x512x256xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x8x8x512xf32>, %arg4: tensor<512xf32>) -> tensor<512x8x8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x8x8x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x8x8x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x8x8x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x8x8x512xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,8,8]{1,3,2,0}"} : (tensor<512x8x8x512xf32>) -> tensor<512x512x8x8xf32>
    %9 = mhlo.copy %8 : tensor<512x512x8x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512x8x8xf32>
    %11 = stablehlo.negate %9 : tensor<512x512x8x8xf32>
    %12 = stablehlo.exponential %11 : tensor<512x512x8x8xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x512x8x8xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x512x8x8xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x512x8x8xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,8,8,512]{2,1,3,0}"} : (tensor<512x512x8x8xf32>) -> tensor<512x8x8x512xf32>
    %17 = mhlo.copy %16 : tensor<512x8x8x512xf32>
    return %17 : tensor<512x8x8x512xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x512xf32>
    return %1 : tensor<3x3x1x512xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x8x8x512xf32>, %arg4: tensor<512xf32>) -> tensor<512x8x8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x8x8x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x8x8x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x8x8x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x8x8x512xf32>
    return %7 : tensor<512x8x8x512xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,768,512]{1,0,2,3}"} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x768x512xf32>
    return %1 : tensor<1x1x768x512xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<512x8x8x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x8x8x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x8x8x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x8x8x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x8x8x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x8x8x768xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,768,8,8]{1,3,2,0}"} : (tensor<512x8x8x768xf32>) -> tensor<512x768x8x8xf32>
    %9 = mhlo.copy %8 : tensor<512x768x8x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x768x8x8xf32>
    %11 = stablehlo.negate %9 : tensor<512x768x8x8xf32>
    %12 = stablehlo.exponential %11 : tensor<512x768x8x8xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x768x8x8xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x768x8x8xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x768x8x8xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,8,8,768]{2,1,3,0}"} : (tensor<512x768x8x8xf32>) -> tensor<512x8x8x768xf32>
    %17 = mhlo.copy %16 : tensor<512x8x8x768xf32>
    return %17 : tensor<512x8x8x768xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,768]{1,0,2,3}"} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x768xf32>
    return %1 : tensor<3x3x1x768xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<512x16x16x768xf32>, %arg4: tensor<768xf32>) -> tensor<512x16x16x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16x16x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x16x16x768xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x16x16x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x16x16x768xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,768,16,16]{1,3,2,0}"} : (tensor<512x16x16x768xf32>) -> tensor<512x768x16x16xf32>
    %9 = mhlo.copy %8 : tensor<512x768x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x768x16x16xf32>
    %11 = stablehlo.negate %9 : tensor<512x768x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<512x768x16x16xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x768x16x16xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x768x16x16xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x768x16x16xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,768]{2,1,3,0}"} : (tensor<512x768x16x16xf32>) -> tensor<512x16x16x768xf32>
    %17 = mhlo.copy %16 : tensor<512x16x16x768xf32>
    return %17 : tensor<512x16x16x768xf32>
  }
  func.func private @fused_computation.118(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,768]{1,0,2,3}"} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x768xf32>
    return %1 : tensor<1x1x384x768xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<512x16x16x384xf32>, %arg4: tensor<384xf32>) -> tensor<512x16x16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16x16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x16x16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x16x16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x16x16x384xf32>
    return %7 : tensor<512x16x16x384xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<512x192x16x16xf32>) -> tensor<512x16x16x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,192]{2,1,3,0}"} : (tensor<512x192x16x16xf32>) -> tensor<512x16x16x192xf32>
    %1 = mhlo.copy %0 : tensor<512x16x16x192xf32>
    return %1 : tensor<512x16x16x192xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<512x4x64x192xf32>, %arg5: tensor<192xf32>, %arg6: tensor<192xf32>) -> tensor<256x512x192x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %2 = stablehlo.add %arg4, %1 : tensor<512x4x64x192xf32>
    %3 = stablehlo.transpose %2, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %4 = mhlo.copy %3 : tensor<512x192x4x64xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<512x192x4x64xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<512xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<512x49152xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<512x49152xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<512x192x4x64xf32>
    %16 = stablehlo.add %0, %15 : tensor<512x192x4x64xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x192x2x2x8x8xf32>
    %18 = stablehlo.transpose %17, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,8,2,8,512,192]{3,1,2,0,5,4}"} : (tensor<512x192x2x2x8x8xf32>) -> tensor<2x8x2x8x512x192xf32>
    %19 = mhlo.copy %18 : tensor<2x8x2x8x512x192xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[256,512,192,1,1]{2,1,0,4,3}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<256x512x192x1x1xf32>
    %21 = mhlo.copy %20 : tensor<256x512x192x1x1xf32>
    return %21 : tensor<256x512x192x1x1xf32>
  }
  func.func private @region_27.35.clone.7.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.125(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>, %arg2: tensor<512x4x64x192xf32>, %arg3: tensor<192xf32>) -> tensor<512x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<512x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<512xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %9 = stablehlo.subtract %5, %8 : tensor<512x49152xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<512x49152xf32>
    return %10 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<1x512x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    return %5 : tensor<1x512x49152xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.129(%arg0: tensor<512x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<512x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<512x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<512x4x64x384xf32>
    return %11 : tensor<512x4x64x384xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.6.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.132(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    return %4 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<512x192x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<512x4x64x192xf32>
    return %10 : tensor<512x4x64x192xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x192x4x64xf32>
    return %7 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x64xf32>
    return %2 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x64xf32>
    return %10 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.140(%arg0: tensor<512x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    return %4 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.5.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.143(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    return %4 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<512x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<512x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<512x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<512x4x64x384xf32>
    return %11 : tensor<512x4x64x384xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.4.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.150(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.151(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    return %4 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<512x192x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<512x4x64x192xf32>
    return %10 : tensor<512x4x64x192xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x192x4x64xf32>
    return %7 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x64xf32>
    return %2 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x64xf32>
    return %10 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<512x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    return %4 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.161(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.162(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    return %4 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<512x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<512x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<512x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<512x4x64x384xf32>
    return %11 : tensor<512x4x64x384xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.2.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.168(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    return %4 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<512x192x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<512x4x64x192xf32>
    return %10 : tensor<512x4x64x192xf32>
  }
  func.func private @fused_computation.173(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x192x4x64xf32>
    return %7 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x64xf32>
    return %2 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x64xf32>
    return %10 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<512x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    return %4 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.179(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<512x4x64x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x192x4x64xf32>
    return %4 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<512x4x64x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x4x64x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x384xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x384x4x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
    %5 = stablehlo.negate %3 : tensor<512x384x4x64xf32>
    %6 = stablehlo.exponential %5 : tensor<512x384x4x64xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x384x4x64xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x384x4x64xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x384x4x64xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
    %11 = mhlo.copy %10 : tensor<512x4x64x384xf32>
    return %11 : tensor<512x4x64x384xf32>
  }
  func.func private @fused_computation.184(%arg0: tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x384xf32>
    return %1 : tensor<1x1x192x384xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x192x4x64xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x49152xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x49152xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x192x4x64xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x192x4x64xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %13 = mhlo.copy %12 : tensor<512x4x64x192xf32>
    return %13 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.186(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<512xf32>, %arg1: tensor<512x192x4x64xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x49152xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x49152xf32>
    return %5 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<512x4x64x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<256x512x192x1x1xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x192xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x192x4x64xf32>
    %4 = mhlo.bitcast %arg2 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
    %5 = stablehlo.transpose %4, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
    %6 = mhlo.copy %5 : tensor<512x192x2x2x8x8xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<512x192x4x64xf32>
    %8 = stablehlo.add %3, %7 : tensor<512x192x4x64xf32>
    return %8 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x192xf32>
    return %1 : tensor<1x1x192x192xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<512x192x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x192x4x64xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x192x4x64xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %10 = mhlo.copy %9 : tensor<512x4x64x192xf32>
    return %10 : tensor<512x4x64x192xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x192x4x64xf32>
    return %7 : tensor<512x192x4x64xf32>
  }
  func.func private @fused_computation.192(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x64xf32>
    return %2 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x64x385xf32>, %arg2: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x64xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x64xf32>
    return %10 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<512x4x64x385xf32>, %arg1: tensor<385xf32>) -> tensor<512x1x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x64x385xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
    %3 = mhlo.copy %2 : tensor<512x385x4x64xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
    return %4 : tensor<512x1x4x64xf32>
  }
  func.func private @fused_computation.195(%arg0: tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %1 = mhlo.copy %0 : tensor<1x1x192x385xf32>
    return %1 : tensor<1x1x192x385xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<192xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<256x512x192x1x1xf32>, %arg4: tensor<192xf32>) -> tensor<512x4x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
    %2 = stablehlo.transpose %1, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
    %3 = mhlo.copy %2 : tensor<512x192x2x2x8x8xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6 = stablehlo.multiply %arg2, %5 : tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<512x49152xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x49152xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<512x192x4x64xf32>
    %14 = stablehlo.add %0, %13 : tensor<512x192x4x64xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
    %16 = mhlo.copy %15 : tensor<512x4x64x192xf32>
    return %16 : tensor<512x4x64x192xf32>
  }
  func.func private @region_27.35.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.197(%arg0: tensor<512x2xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<512xf32>, %arg1: tensor<256x512x192x1x1xf32>) -> tensor<512x49152xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
    %2 = mhlo.copy %1 : tensor<512x192x2x2x8x8xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<512x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
    %7 = stablehlo.subtract %3, %6 : tensor<512x49152xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x49152xf32>
    return %8 : tensor<512x49152xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<256x512x192x1x1xf32>) -> tensor<1x512x49152xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
    %2 = mhlo.copy %1 : tensor<512x192x2x2x8x8xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<1x512x49152xf32>
    return %3 : tensor<1x512x49152xf32>
  }
  func.func private @fused_computation.200(%arg0: tensor<512x16x16x192xf32>, %arg1: tensor<2x8x2x8x2xi32>) -> tensor<256x512x192x1x1xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,16,16]{1,3,2,0}"} : (tensor<512x16x16x192xf32>) -> tensor<512x192x16x16xf32>
    %1 = mhlo.copy %0 : tensor<512x192x16x16xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    %3 = "stablehlo.gather"(%1, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 512, 192, 1, 1>}> : (tensor<512x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x512x192x1x1xf32>
    return %3 : tensor<256x512x192x1x1xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %1 = mhlo.copy %0 : tensor<1x1x384x192xf32>
    return %1 : tensor<1x1x384x192xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<512x16x16x384xf32>, %arg4: tensor<384xf32>) -> tensor<512x16x16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16x16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x16x16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x16x16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x16x16x384xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,16,16]{1,3,2,0}"} : (tensor<512x16x16x384xf32>) -> tensor<512x384x16x16xf32>
    %9 = mhlo.copy %8 : tensor<512x384x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x384x16x16xf32>
    %11 = stablehlo.negate %9 : tensor<512x384x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<512x384x16x16xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x384x16x16xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x384x16x16xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x384x16x16xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,384]{2,1,3,0}"} : (tensor<512x384x16x16xf32>) -> tensor<512x16x16x384xf32>
    %17 = mhlo.copy %16 : tensor<512x16x16x384xf32>
    return %17 : tensor<512x16x16x384xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,384]{1,0,2,3}"} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x384xf32>
    return %1 : tensor<3x3x1x384xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<512x16x16x384xf32>, %arg4: tensor<384xf32>) -> tensor<512x16x16x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16x16x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x16x16x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x16x16x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x16x16x384xf32>
    return %7 : tensor<512x16x16x384xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_computation.207(%arg0: tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,384]{1,0,2,3}"} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
    %1 = mhlo.copy %0 : tensor<1x1x512x384xf32>
    return %1 : tensor<1x1x512x384xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x16x16x512xf32>, %arg4: tensor<512xf32>) -> tensor<512x16x16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x16x16x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x16x16x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x16x16x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x16x16x512xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,16,16]{1,3,2,0}"} : (tensor<512x16x16x512xf32>) -> tensor<512x512x16x16xf32>
    %9 = mhlo.copy %8 : tensor<512x512x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512x16x16xf32>
    %11 = stablehlo.negate %9 : tensor<512x512x16x16xf32>
    %12 = stablehlo.exponential %11 : tensor<512x512x16x16xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x512x16x16xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x512x16x16xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x512x16x16xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,512]{2,1,3,0}"} : (tensor<512x512x16x16xf32>) -> tensor<512x16x16x512xf32>
    %17 = mhlo.copy %16 : tensor<512x16x16x512xf32>
    return %17 : tensor<512x16x16x512xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.210(%arg0: tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x512xf32>
    return %1 : tensor<3x3x1x512xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x32x32x512xf32>, %arg4: tensor<512xf32>) -> tensor<512x32x32x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x32x32x512xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x32x32x512xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x32x32x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x32x32x512xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,32,32]{1,3,2,0}"} : (tensor<512x32x32x512xf32>) -> tensor<512x512x32x32xf32>
    %9 = mhlo.copy %8 : tensor<512x512x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x512x32x32xf32>
    %11 = stablehlo.negate %9 : tensor<512x512x32x32xf32>
    %12 = stablehlo.exponential %11 : tensor<512x512x32x32xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x512x32x32xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x512x32x32xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x512x32x32xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,512]{2,1,3,0}"} : (tensor<512x512x32x32xf32>) -> tensor<512x32x32x512xf32>
    %17 = mhlo.copy %16 : tensor<512x32x32x512xf32>
    return %17 : tensor<512x32x32x512xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x32x32x256xf32>
    return %7 : tensor<512x32x32x256xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<512x128x32x32xf32>) -> tensor<512x32x32x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,128]{2,1,3,0}"} : (tensor<512x128x32x32xf32>) -> tensor<512x32x32x128xf32>
    %1 = mhlo.copy %0 : tensor<512x32x32x128xf32>
    return %1 : tensor<512x32x32x128xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<128xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x128x4x256xf32>, %arg4: tensor<512x4x256x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>) -> tensor<1024x512x128x1x1xf32> {
    %0 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %2 = stablehlo.add %arg4, %1 : tensor<512x4x256x128xf32>
    %3 = stablehlo.transpose %2, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
    %4 = mhlo.copy %3 : tensor<512x128x4x256xf32>
    %5 = stablehlo.add %4, %arg3 : tensor<512x128x4x256xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %8 = stablehlo.multiply %arg2, %7 : tensor<512xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %10 = stablehlo.subtract %6, %9 : tensor<512x131072xf32>
    %11 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<512x131072xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<512x128x4x256xf32>
    %16 = stablehlo.add %0, %15 : tensor<512x128x4x256xf32>
    %17 = mhlo.bitcast %16 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x128x2x2x16x16xf32>
    %18 = stablehlo.transpose %17, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,16,2,16,512,128]{3,1,2,0,5,4}"} : (tensor<512x128x2x2x16x16xf32>) -> tensor<2x16x2x16x512x128xf32>
    %19 = mhlo.copy %18 : tensor<2x16x2x16x512x128xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1024,512,128,1,1]{2,1,0,4,3}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<1024x512x128x1x1xf32>
    %21 = mhlo.copy %20 : tensor<1024x512x128x1x1xf32>
    return %21 : tensor<1024x512x128x1x1xf32>
  }
  func.func private @region_3.3.clone.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.219(%arg0: tensor<512x4xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.220(%arg0: tensor<512xf32>, %arg1: tensor<512x128x4x256xf32>, %arg2: tensor<512x4x256x128xf32>, %arg3: tensor<128xf32>) -> tensor<512x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<512x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x128x4x256xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<512x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<512xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %9 = stablehlo.subtract %5, %8 : tensor<512x131072xf32>
    %10 = stablehlo.multiply %9, %9 : tensor<512x131072xf32>
    return %10 : tensor<512x131072xf32>
  }
  func.func private @fused_computation.221(%arg0: tensor<512x128x4x256xf32>, %arg1: tensor<512x4x256x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x512x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x128x4x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<1x512x131072xf32>
    return %5 : tensor<1x512x131072xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<512x4x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x4x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x4x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x256x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,256]{1,3,2,0}"} : (tensor<512x4x256x256xf32>) -> tensor<512x256x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x256xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x4x256xf32>
    %5 = stablehlo.negate %3 : tensor<512x256x4x256xf32>
    %6 = stablehlo.exponential %5 : tensor<512x256x4x256xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x256x4x256xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x256x4x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x256x4x256xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,256]{2,1,3,0}"} : (tensor<512x256x4x256xf32>) -> tensor<512x4x256x256xf32>
    %11 = mhlo.copy %10 : tensor<512x4x256x256xf32>
    return %11 : tensor<512x4x256x256xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.225(%arg0: tensor<128xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x128x4x256xf32>, %arg4: tensor<128xf32>) -> tensor<512x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x131072xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x131072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x128x4x256xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x128x4x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
    %13 = mhlo.copy %12 : tensor<512x4x256x128xf32>
    return %13 : tensor<512x4x256x128xf32>
  }
  func.func private @region_3.3.clone.2.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.226(%arg0: tensor<512x4xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<512xf32>, %arg1: tensor<512x128x4x256xf32>) -> tensor<512x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x131072xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x131072xf32>
    return %5 : tensor<512x131072xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<512x128x4x256xf32>, %arg1: tensor<512x4x256x128xf32>, %arg2: tensor<128xf32>) -> tensor<512x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x128x4x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x128x4x256xf32>
    return %4 : tensor<512x128x4x256xf32>
  }
  func.func private @fused_computation.229(%arg0: tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x128xf32>
    return %1 : tensor<1x1x128x128xf32>
  }
  func.func private @fused_computation.230(%arg0: tensor<512x128x4xf32>, %arg1: tensor<512x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<512x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 129:257, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x128x4x256xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x128x4x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x128x4xf32>) -> tensor<512x128x4x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x128x4x256xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
    %10 = mhlo.copy %9 : tensor<512x4x256x128xf32>
    return %10 : tensor<512x4x256x128xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<512x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<512x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 1:129, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x256xf32>) -> tensor<512x4x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x256xf32>) -> tensor<512x128x4x256xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x128x4x256xf32>
    return %7 : tensor<512x128x4x256xf32>
  }
  func.func private @fused_computation.232(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x256xf32>
    return %2 : tensor<512x1x4x256xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<512x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x256xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x256xf32>
    return %10 : tensor<512x1x4x256xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<512x4x256x257xf32>, %arg1: tensor<257xf32>) -> tensor<512x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
    return %4 : tensor<512x1x4x256xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x257xf32>
    return %1 : tensor<1x1x128x257xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<128xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x128x4x256xf32>, %arg4: tensor<128xf32>) -> tensor<512x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x131072xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x131072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x128x4x256xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x128x4x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
    %13 = mhlo.copy %12 : tensor<512x4x256x128xf32>
    return %13 : tensor<512x4x256x128xf32>
  }
  func.func private @region_3.3.clone.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.237(%arg0: tensor<512x4xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.238(%arg0: tensor<512xf32>, %arg1: tensor<512x128x4x256xf32>) -> tensor<512x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x131072xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x131072xf32>
    return %5 : tensor<512x131072xf32>
  }
  func.func private @fused_computation.239(%arg0: tensor<512x128x4x256xf32>, %arg1: tensor<512x4x256x128xf32>, %arg2: tensor<128xf32>) -> tensor<512x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x128x4x256xf32>
    %4 = stablehlo.add %3, %arg0 : tensor<512x128x4x256xf32>
    return %4 : tensor<512x128x4x256xf32>
  }
  func.func private @fused_computation.240(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.241(%arg0: tensor<512x4x256x256xf32>, %arg1: tensor<256xf32>) -> tensor<512x4x256x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x4x256x256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x256x256xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,256]{1,3,2,0}"} : (tensor<512x4x256x256xf32>) -> tensor<512x256x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x256x4x256xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x4x256xf32>
    %5 = stablehlo.negate %3 : tensor<512x256x4x256xf32>
    %6 = stablehlo.exponential %5 : tensor<512x256x4x256xf32>
    %7 = stablehlo.add %4, %6 : tensor<512x256x4x256xf32>
    %8 = stablehlo.divide %4, %7 : tensor<512x256x4x256xf32>
    %9 = stablehlo.multiply %3, %8 : tensor<512x256x4x256xf32>
    %10 = stablehlo.transpose %9, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,256]{2,1,3,0}"} : (tensor<512x256x4x256xf32>) -> tensor<512x4x256x256xf32>
    %11 = mhlo.copy %10 : tensor<512x4x256x256xf32>
    return %11 : tensor<512x4x256x256xf32>
  }
  func.func private @fused_computation.242(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.243(%arg0: tensor<128xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x128x4x256xf32>, %arg4: tensor<128xf32>) -> tensor<512x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg2, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x131072xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x131072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x128x4x256xf32>
    %11 = stablehlo.add %0, %10 : tensor<512x128x4x256xf32>
    %12 = stablehlo.transpose %11, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
    %13 = mhlo.copy %12 : tensor<512x4x256x128xf32>
    return %13 : tensor<512x4x256x128xf32>
  }
  func.func private @region_3.3.clone.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.244(%arg0: tensor<512x4xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.245(%arg0: tensor<512xf32>, %arg1: tensor<512x128x4x256xf32>) -> tensor<512x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<512x131072xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<512x131072xf32>
    return %5 : tensor<512x131072xf32>
  }
  func.func private @fused_computation.246(%arg0: tensor<512x4x256x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1024x512x128x1x1xf32>) -> tensor<512x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x256x128xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x128x4x256xf32>
    %4 = mhlo.bitcast %arg2 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
    %5 = stablehlo.transpose %4, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
    %6 = mhlo.copy %5 : tensor<512x128x2x2x16x16xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<512x128x4x256xf32>
    %8 = stablehlo.add %3, %7 : tensor<512x128x4x256xf32>
    return %8 : tensor<512x128x4x256xf32>
  }
  func.func private @fused_computation.247(%arg0: tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x128xf32>
    return %1 : tensor<1x1x128x128xf32>
  }
  func.func private @fused_computation.248(%arg0: tensor<512x128x4xf32>, %arg1: tensor<512x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<512x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 129:257, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x128x4x256xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<512x128x4x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<512x128x4xf32>) -> tensor<512x128x4x256xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<512x128x4x256xf32>
    %9 = stablehlo.transpose %8, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
    %10 = mhlo.copy %9 : tensor<512x4x256x128xf32>
    return %10 : tensor<512x4x256x128xf32>
  }
  func.func private @fused_computation.249(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<512x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<512x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 1:129, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
    %5 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x256xf32>) -> tensor<512x4x256xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 2, 3] : (tensor<512x4x256xf32>) -> tensor<512x128x4x256xf32>
    %7 = stablehlo.multiply %4, %6 : tensor<512x128x4x256xf32>
    return %7 : tensor<512x128x4x256xf32>
  }
  func.func private @fused_computation.250(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<512x1x4xf32>) -> tensor<512x1x4x256xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
    %2 = stablehlo.divide %arg0, %1 : tensor<512x1x4x256xf32>
    return %2 : tensor<512x1x4x256xf32>
  }
  func.func private @fused_computation.251(%arg0: tensor<512x1x4xf32>, %arg1: tensor<512x4x256x257xf32>, %arg2: tensor<257xf32>) -> tensor<512x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
    %6 = stablehlo.maximum %5, %arg0 : tensor<512x1x4xf32>
    %7 = mhlo.bitcast %6 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
    %9 = stablehlo.subtract %4, %8 : tensor<512x1x4x256xf32>
    %10 = stablehlo.exponential %9 : tensor<512x1x4x256xf32>
    return %10 : tensor<512x1x4x256xf32>
  }
  func.func private @fused_computation.252(%arg0: tensor<512x4x256x257xf32>, %arg1: tensor<257xf32>) -> tensor<512x1x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x4x256x257xf32>
    %2 = stablehlo.transpose %1, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
    %3 = mhlo.copy %2 : tensor<512x257x4x256xf32>
    %4 = stablehlo.slice %3 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
    return %4 : tensor<512x1x4x256xf32>
  }
  func.func private @fused_computation.253(%arg0: tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x257xf32>
    return %1 : tensor<1x1x128x257xf32>
  }
  func.func private @fused_computation.254(%arg0: tensor<128xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1024x512x128x1x1xf32>, %arg4: tensor<128xf32>) -> tensor<512x4x256x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %1 = mhlo.bitcast %arg3 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
    %2 = stablehlo.transpose %1, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
    %3 = mhlo.copy %2 : tensor<512x128x2x2x16x16xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %6 = stablehlo.multiply %arg2, %5 : tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<512x131072xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<512x131072xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<512x128x4x256xf32>
    %14 = stablehlo.add %0, %13 : tensor<512x128x4x256xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
    %16 = mhlo.copy %15 : tensor<512x4x256x128xf32>
    return %16 : tensor<512x4x256x128xf32>
  }
  func.func private @region_3.3.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.255(%arg0: tensor<512x4xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.256(%arg0: tensor<512xf32>, %arg1: tensor<1024x512x128x1x1xf32>) -> tensor<512x131072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
    %2 = mhlo.copy %1 : tensor<512x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<512x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
    %7 = stablehlo.subtract %3, %6 : tensor<512x131072xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<512x131072xf32>
    return %8 : tensor<512x131072xf32>
  }
  func.func private @fused_computation.257(%arg0: tensor<1024x512x128x1x1xf32>) -> tensor<1x512x131072xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
    %1 = stablehlo.transpose %0, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
    %2 = mhlo.copy %1 : tensor<512x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<1x512x131072xf32>
    return %3 : tensor<1x512x131072xf32>
  }
  func.func private @fused_computation.258(%arg0: tensor<512x32x32x128xf32>, %arg1: tensor<2x16x2x16x2xi32>) -> tensor<1024x512x128x1x1xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,32,32]{1,3,2,0}"} : (tensor<512x32x32x128xf32>) -> tensor<512x128x32x32xf32>
    %1 = mhlo.copy %0 : tensor<512x128x32x32xf32>
    %2 = mhlo.bitcast %arg1 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    %3 = "stablehlo.gather"(%1, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 512, 128, 1, 1>}> : (tensor<512x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x512x128x1x1xf32>
    return %3 : tensor<1024x512x128x1x1xf32>
  }
  func.func private @fused_computation.259(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.260(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x32x32x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,32,32]{1,3,2,0}"} : (tensor<512x32x32x256xf32>) -> tensor<512x256x32x32xf32>
    %9 = mhlo.copy %8 : tensor<512x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x32x32xf32>
    %11 = stablehlo.negate %9 : tensor<512x256x32x32xf32>
    %12 = stablehlo.exponential %11 : tensor<512x256x32x32xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x256x32x32xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x256x32x32xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x256x32x32xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,256]{2,1,3,0}"} : (tensor<512x256x32x32xf32>) -> tensor<512x32x32x256xf32>
    %17 = mhlo.copy %16 : tensor<512x32x32x256xf32>
    return %17 : tensor<512x32x32x256xf32>
  }
  func.func private @fused_computation.261(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.262(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.263(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x32x32x256xf32>
    return %7 : tensor<512x32x32x256xf32>
  }
  func.func private @fused_computation.264(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.265(%arg0: tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
  func.func private @fused_computation.266(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x32x32x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x32x32x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x32x32x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x32x32x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x32x32x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x32x32x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,32,32]{1,3,2,0}"} : (tensor<512x32x32x256xf32>) -> tensor<512x256x32x32xf32>
    %9 = mhlo.copy %8 : tensor<512x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x32x32xf32>
    %11 = stablehlo.negate %9 : tensor<512x256x32x32xf32>
    %12 = stablehlo.exponential %11 : tensor<512x256x32x32xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x256x32x32xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x256x32x32xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x256x32x32xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,256]{2,1,3,0}"} : (tensor<512x256x32x32xf32>) -> tensor<512x32x32x256xf32>
    %17 = mhlo.copy %16 : tensor<512x32x32x256xf32>
    return %17 : tensor<512x32x32x256xf32>
  }
  func.func private @fused_computation.267(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.268(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.269(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x64x64x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x64x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x64x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,64,64]{1,3,2,0}"} : (tensor<512x64x64x256xf32>) -> tensor<512x256x64x64xf32>
    %9 = mhlo.copy %8 : tensor<512x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<512x256x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<512x256x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x256x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x256x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x256x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,256]{2,1,3,0}"} : (tensor<512x256x64x64xf32>) -> tensor<512x64x64x256xf32>
    %17 = mhlo.copy %16 : tensor<512x64x64x256xf32>
    return %17 : tensor<512x64x64x256xf32>
  }
  func.func private @fused_computation.270(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.271(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.272(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<512x64x64x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<512x64x64x128xf32>, %arg9: tensor<128xf32>) -> tensor<512x64x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg9, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %1 = stablehlo.subtract %arg8, %0 : tensor<512x64x64x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x64x64x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg6, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x64x64x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x64x64x128xf32>
    %8 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %9 = stablehlo.subtract %arg3, %8 : tensor<512x64x64x128xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<512x64x64x128xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<512x64x64x128xf32>
    %14 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %15 = stablehlo.add %13, %14 : tensor<512x64x64x128xf32>
    %16 = stablehlo.add %7, %15 : tensor<512x64x64x128xf32>
    return %16 : tensor<512x64x64x128xf32>
  }
  func.func private @fused_computation.273(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.274(%arg0: tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x256x128xf32>
    return %1 : tensor<1x1x256x128xf32>
  }
  func.func private @fused_computation.275(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x64x64x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x64x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x64x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,64,64]{1,3,2,0}"} : (tensor<512x64x64x256xf32>) -> tensor<512x256x64x64xf32>
    %9 = mhlo.copy %8 : tensor<512x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<512x256x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<512x256x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x256x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x256x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x256x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,256]{2,1,3,0}"} : (tensor<512x256x64x64xf32>) -> tensor<512x64x64x256xf32>
    %17 = mhlo.copy %16 : tensor<512x64x64x256xf32>
    return %17 : tensor<512x64x64x256xf32>
  }
  func.func private @fused_computation.276(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.277(%arg0: tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x256xf32>
    return %1 : tensor<3x3x1x256xf32>
  }
  func.func private @fused_computation.278(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<512x64x64x256xf32>, %arg4: tensor<256xf32>) -> tensor<512x64x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x64x64x256xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x64x64x256xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x64x64x256xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x64x64x256xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,64,64]{1,3,2,0}"} : (tensor<512x64x64x256xf32>) -> tensor<512x256x64x64xf32>
    %9 = mhlo.copy %8 : tensor<512x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x256x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<512x256x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<512x256x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x256x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x256x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x256x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,256]{2,1,3,0}"} : (tensor<512x256x64x64xf32>) -> tensor<512x64x64x256xf32>
    %17 = mhlo.copy %16 : tensor<512x64x64x256xf32>
    return %17 : tensor<512x64x64x256xf32>
  }
  func.func private @fused_computation.279(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_computation.280(%arg0: tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x256xf32>
    return %1 : tensor<1x1x128x256xf32>
  }
  func.func private @fused_computation.281(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<512x64x64x128xf32>, %arg4: tensor<128xf32>) -> tensor<512x64x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x64x64x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x64x64x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x64x64x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x64x64x128xf32>
    return %7 : tensor<512x64x64x128xf32>
  }
  func.func private @fused_computation.282(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.283(%arg0: tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x128x128xf32>
    return %1 : tensor<1x1x128x128xf32>
  }
  func.func private @fused_computation.284(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<512x64x64x128xf32>, %arg4: tensor<128xf32>) -> tensor<512x64x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x64x64x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x64x64x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x64x64x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x64x64x128xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,64,64]{1,3,2,0}"} : (tensor<512x64x64x128xf32>) -> tensor<512x128x64x64xf32>
    %9 = mhlo.copy %8 : tensor<512x128x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x128x64x64xf32>
    %11 = stablehlo.negate %9 : tensor<512x128x64x64xf32>
    %12 = stablehlo.exponential %11 : tensor<512x128x64x64xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x128x64x64xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x128x64x64xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x128x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,128]{2,1,3,0}"} : (tensor<512x128x64x64xf32>) -> tensor<512x64x64x128xf32>
    %17 = mhlo.copy %16 : tensor<512x64x64x128xf32>
    return %17 : tensor<512x64x64x128xf32>
  }
  func.func private @fused_computation.285(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.286(%arg0: tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x128xf32>
    return %1 : tensor<3x3x1x128xf32>
  }
  func.func private @fused_computation.287(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<512x128x128x128xf32>, %arg4: tensor<128xf32>) -> tensor<512x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x128x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x128x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x128x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x128x128x128xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,128]{1,3,2,0}"} : (tensor<512x128x128x128xf32>) -> tensor<512x128x128x128xf32>
    %9 = mhlo.copy %8 : tensor<512x128x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x128x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<512x128x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<512x128x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x128x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x128x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x128x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,128]{2,1,3,0}"} : (tensor<512x128x128x128xf32>) -> tensor<512x128x128x128xf32>
    %17 = mhlo.copy %16 : tensor<512x128x128x128xf32>
    return %17 : tensor<512x128x128x128xf32>
  }
  func.func private @fused_computation.288(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_computation.289(%arg0: tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x128xf32>
    return %1 : tensor<1x1x64x128xf32>
  }
  func.func private @fused_computation.290(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<512x128x128x64xf32>, %arg4: tensor<64xf32>) -> tensor<512x128x128x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x128x128x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x128x128x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x128x128x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x128x128x64xf32>
    return %7 : tensor<512x128x128x64xf32>
  }
  func.func private @fused_computation.291(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.292(%arg0: tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x64x64xf32>
    return %1 : tensor<1x1x64x64xf32>
  }
  func.func private @fused_computation.293(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<512x128x128x64xf32>, %arg4: tensor<64xf32>) -> tensor<512x128x128x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x128x128x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x128x128x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x128x128x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x128x128x64xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,128,128]{1,3,2,0}"} : (tensor<512x128x128x64xf32>) -> tensor<512x64x128x128xf32>
    %9 = mhlo.copy %8 : tensor<512x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x64x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<512x64x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<512x64x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x64x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x64x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x64x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,64]{2,1,3,0}"} : (tensor<512x64x128x128xf32>) -> tensor<512x128x128x64xf32>
    %17 = mhlo.copy %16 : tensor<512x128x128x64xf32>
    return %17 : tensor<512x128x128x64xf32>
  }
  func.func private @fused_computation.294(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.295(%arg0: tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,64]{1,0,2,3}"} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
    %1 = mhlo.copy %0 : tensor<3x3x1x64xf32>
    return %1 : tensor<3x3x1x64xf32>
  }
  func.func private @fused_computation.296(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<512x128x128x64xf32>, %arg4: tensor<64xf32>) -> tensor<512x128x128x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x128x128x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x128x128x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x128x128x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x128x128x64xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,128,128]{1,3,2,0}"} : (tensor<512x128x128x64xf32>) -> tensor<512x64x128x128xf32>
    %9 = mhlo.copy %8 : tensor<512x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x64x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<512x64x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<512x64x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x64x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x64x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x64x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,64]{2,1,3,0}"} : (tensor<512x64x128x128xf32>) -> tensor<512x128x128x64xf32>
    %17 = mhlo.copy %16 : tensor<512x128x128x64xf32>
    return %17 : tensor<512x128x128x64xf32>
  }
  func.func private @fused_computation.297(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_computation.298(%arg0: tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,32,64]{1,0,2,3}"} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
    %1 = mhlo.copy %0 : tensor<1x1x32x64xf32>
    return %1 : tensor<1x1x32x64xf32>
  }
  func.func private @fused_computation.299(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<512x128x128x32xf32>, %arg4: tensor<32xf32>) -> tensor<512x128x128x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<512x128x128x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<512x128x128x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x128x128x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<512x128x128x32xf32>
    %8 = stablehlo.transpose %7, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,128,128]{1,3,2,0}"} : (tensor<512x128x128x32xf32>) -> tensor<512x32x128x128xf32>
    %9 = mhlo.copy %8 : tensor<512x32x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x32x128x128xf32>
    %11 = stablehlo.negate %9 : tensor<512x32x128x128xf32>
    %12 = stablehlo.exponential %11 : tensor<512x32x128x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<512x32x128x128xf32>
    %14 = stablehlo.divide %10, %13 : tensor<512x32x128x128xf32>
    %15 = stablehlo.multiply %9, %14 : tensor<512x32x128x128xf32>
    %16 = stablehlo.transpose %15, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,32]{2,1,3,0}"} : (tensor<512x32x128x128xf32>) -> tensor<512x128x128x32xf32>
    %17 = mhlo.copy %16 : tensor<512x128x128x32xf32>
    return %17 : tensor<512x128x128x32xf32>
  }
  func.func private @fused_computation.300(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_computation.301(%arg0: tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,32]{1,0,2,3}"} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
    %1 = mhlo.copy %0 : tensor<3x3x3x32xf32>
    return %1 : tensor<3x3x3x32xf32>
  }
  func.func private @fused_computation.302(%arg0: tensor<512x3x256x256xf32>) -> tensor<512x256x256x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,256,3]{2,1,3,0}"} : (tensor<512x3x256x256xf32>) -> tensor<512x256x256x3xf32>
    %1 = mhlo.copy %0 : tensor<512x256x256x3xf32>
    return %1 : tensor<512x256x256x3xf32>
  }
  func.func private @fused_computation.303() -> tensor<1024x2xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
    %4 = mhlo.copy %3 : tensor<2x16x1x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %8 = stablehlo.add %4, %7 : tensor<2x16x1x1xi32>
    %9 = stablehlo.select %6, %8, %4 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %13 = stablehlo.compare LT, %2, %12 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %15 = stablehlo.add %2, %14 : tensor<2x16xi32>
    %16 = stablehlo.select %13, %15, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %18 = stablehlo.concatenate %11, %17, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    %19 = mhlo.bitcast %18 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    return %19 : tensor<1024x2xi32>
  }
  func.func private @fused_computation.304() -> tensor<2x16x2x16x2xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
    %4 = mhlo.copy %3 : tensor<2x16x1x1xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %6 = stablehlo.compare LT, %4, %5 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
    %8 = stablehlo.add %4, %7 : tensor<2x16x1x1xi32>
    %9 = stablehlo.select %6, %8, %4 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %13 = stablehlo.compare LT, %2, %12 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %15 = stablehlo.add %2, %14 : tensor<2x16xi32>
    %16 = stablehlo.select %13, %15, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %18 = stablehlo.concatenate %11, %17, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    return %18 : tensor<2x16x2x16x2xi32>
  }
  func.func private @fused_computation.305() -> tensor<256x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x8x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x8x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x8xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    return %22 : tensor<256x2xi32>
  }
  func.func private @fused_computation.306() -> tensor<2x8x2x8x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x8x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x8x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x8xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    return %21 : tensor<2x8x2x8x2xi32>
  }
  func.func private @fused_computation.307() -> tensor<64x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x4x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x4x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x4xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    %22 = mhlo.bitcast %21 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    return %22 : tensor<64x2xi32>
  }
  func.func private @fused_computation.308() -> tensor<2x4x2x4x2xi32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
    %7 = mhlo.copy %6 : tensor<2x4x1x1xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %9 = stablehlo.compare LT, %7, %8 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
    %11 = stablehlo.add %7, %10 : tensor<2x4x1x1xi32>
    %12 = stablehlo.select %9, %11, %7 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %15 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %16 = stablehlo.compare LT, %5, %15 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %18 = stablehlo.add %5, %17 : tensor<2x4xi32>
    %19 = stablehlo.select %16, %18, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %21 = stablehlo.concatenate %14, %20, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    return %21 : tensor<2x4x2x4x2xi32>
  }
  func.func private @wrapped_broadcast_computation(%arg0: tensor<f32>) -> tensor<512x256x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<512x256x8x8xf32>
    return %0 : tensor<512x256x8x8xf32>
  }
  func.func private @wrapped_broadcast_computation.1(%arg0: tensor<f32>) -> tensor<512x192x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<512x192x16x16xf32>
    return %0 : tensor<512x192x16x16xf32>
  }
  func.func private @wrapped_broadcast_computation.2(%arg0: tensor<f32>) -> tensor<512x128x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<512x128x32x32xf32>
    return %0 : tensor<512x128x32x32xf32>
  }
  func.func private @region_1.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<1x512x131072xf32>, %arg1: tensor<f32>) -> tensor<1x512x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    return %0 : tensor<1x512x128xf32>
  }
  func.func private @region_3.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<512x131072xf32>, %arg1: tensor<f32>) -> tensor<512x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    return %0 : tensor<512x4096xf32>
  }
  func.func private @region_5.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<512x1x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4x8xf32>
    return %0 : tensor<512x1x4x8xf32>
  }
  func.func private @region_5.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<512x1x4x8xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x8xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_7.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<512x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<512x128x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x128x4x256xf32>, tensor<f32>) -> tensor<512x128x4x8xf32>
    return %0 : tensor<512x128x4x8xf32>
  }
  func.func private @region_9.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<1x512x131072xf32>, %arg1: tensor<f32>) -> tensor<1x512x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    return %0 : tensor<1x512x128xf32>
  }
  func.func private @region_3.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<512x131072xf32>, %arg1: tensor<f32>) -> tensor<512x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    return %0 : tensor<512x4096xf32>
  }
  func.func private @region_12.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<1x512x131072xf32>, %arg1: tensor<f32>) -> tensor<1x512x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    return %0 : tensor<1x512x128xf32>
  }
  func.func private @region_3.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<512x131072xf32>, %arg1: tensor<f32>) -> tensor<512x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    return %0 : tensor<512x4096xf32>
  }
  func.func private @region_14.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<512x1x4x256xf32>, %arg1: tensor<f32>) -> tensor<512x1x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4x8xf32>
    return %0 : tensor<512x1x4x8xf32>
  }
  func.func private @region_14.21.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<512x1x4x8xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x8xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_16.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<512x128x4x256xf32>, %arg1: tensor<f32>) -> tensor<512x128x4x8xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x128x4x256xf32>, tensor<f32>) -> tensor<512x128x4x8xf32>
    return %0 : tensor<512x128x4x8xf32>
  }
  func.func private @region_18.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<1x512x131072xf32>, %arg1: tensor<f32>) -> tensor<1x512x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    return %0 : tensor<1x512x128xf32>
  }
  func.func private @region_3.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<512x131072xf32>, %arg1: tensor<f32>) -> tensor<512x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    return %0 : tensor<512x4096xf32>
  }
  func.func private @region_21.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<1x512x131072xf32>, %arg1: tensor<f32>) -> tensor<1x512x128xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    return %0 : tensor<1x512x128xf32>
  }
  func.func private @region_3.3.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<512x131072xf32>, %arg1: tensor<f32>) -> tensor<512x4096xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    return %0 : tensor<512x4096xf32>
  }
  func.func private @region_23.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_scatter_computation(%arg0: tensor<512x128x32x32xf32>, %arg1: tensor<1024x2xi32>, %arg2: tensor<1024x512x128x1x1xf32>) -> tensor<512x128x32x32xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x512x128x1x1xf32>) -> tensor<512x128x32x32xf32>
    return %0 : tensor<512x128x32x32xf32>
  }
  func.func private @region_25.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_25.33.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_25.33.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_29.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    return %0 : tensor<512x1x4x2xf32>
  }
  func.func private @region_29.40.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<512x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_31.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    return %0 : tensor<512x192x4x2xf32>
  }
  func.func private @region_33.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_33.44.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_33.44.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_36.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_36.48.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_36.48.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_38.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    return %0 : tensor<512x1x4x2xf32>
  }
  func.func private @region_38.52.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<512x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_40.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    return %0 : tensor<512x192x4x2xf32>
  }
  func.func private @region_42.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_42.55.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_42.55.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_45.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_45.59.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_45.59.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_47.63(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.33(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    return %0 : tensor<512x1x4x2xf32>
  }
  func.func private @region_47.63.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<512x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_49.65(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.34(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    return %0 : tensor<512x192x4x2xf32>
  }
  func.func private @region_51.66(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.35(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_51.66.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.36(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_51.66.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.37(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_54.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.38(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_54.70.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.39(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_54.70.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.40(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_56.74(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.41(%arg0: tensor<512x1x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x1x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    return %0 : tensor<512x1x4x2xf32>
  }
  func.func private @region_56.74.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<512x1x4x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    return %0 : tensor<512x1x4xf32>
  }
  func.func private @region_58.76(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.42(%arg0: tensor<512x192x4x64xf32>, %arg1: tensor<f32>) -> tensor<512x192x4x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    return %0 : tensor<512x192x4x2xf32>
  }
  func.func private @region_60.77(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.43(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_60.77.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.44(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_60.77.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.45(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_63.81(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.46(%arg0: tensor<1x512x49152xf32>, %arg1: tensor<f32>) -> tensor<1x512x48xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    return %0 : tensor<1x512x48xf32>
  }
  func.func private @region_63.81.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.47(%arg0: tensor<1x512x48xf32>, %arg1: tensor<f32>) -> tensor<1x512x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    return %0 : tensor<1x512x2xf32>
  }
  func.func private @region_63.81.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<1x512x2xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_27.35.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.48(%arg0: tensor<512x49152xf32>, %arg1: tensor<f32>) -> tensor<512x1536xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    return %0 : tensor<512x1536xf32>
  }
  func.func private @region_65.85(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_scatter_computation.1(%arg0: tensor<512x192x16x16xf32>, %arg1: tensor<256x2xi32>, %arg2: tensor<256x512x192x1x1xf32>) -> tensor<512x192x16x16xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x192x16x16xf32>, tensor<256x2xi32>, tensor<256x512x192x1x1xf32>) -> tensor<512x192x16x16xf32>
    return %0 : tensor<512x192x16x16xf32>
  }
  func.func private @region_67.86(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.49(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_67.86.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.15(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.50(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_75.97(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.51(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_75.97.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.16(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.52(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_78.101(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.53(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_78.101.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.17(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.54(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_84.108(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.55(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_84.108.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.18(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.56(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_87.112(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.57(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_87.112.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.19(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.58(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_93.119(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.59(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_93.119.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.20(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.60(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_96.123(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.61(%arg0: tensor<1x512x16384xf32>, %arg1: tensor<f32>) -> tensor<1x512x16xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    return %0 : tensor<1x512x16xf32>
  }
  func.func private @region_96.123.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.21(%arg0: tensor<1x512x16xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.88.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.62(%arg0: tensor<512x16384xf32>, %arg1: tensor<f32>) -> tensor<512x512xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  func.func private @region_98.127(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_scatter_computation.2(%arg0: tensor<512x256x8x8xf32>, %arg1: tensor<64x2xi32>, %arg2: tensor<64x512x256x1x1xf32>) -> tensor<512x256x8x8xf32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x256x8x8xf32>, tensor<64x2xi32>, tensor<64x512x256x1x1xf32>) -> tensor<512x256x8x8xf32>
    return %0 : tensor<512x256x8x8xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<512x3x256x256xf32>) -> tensor<512x2xf32> {
    %0 = "mhlo.fusion"(%arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512xf32>
      %644 = stablehlo.rsqrt %643 : tensor<512xf32>
      mhlo.return %644 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %642 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<512x256x8x8xf32>
      mhlo.return %642 : tensor<512x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<512x256x8x8xf32>
    %2 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %642 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %643 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %644 = stablehlo.multiply %642, %643 : tensor<4xi32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %646 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %647 = stablehlo.add %645, %646 : tensor<2x4xi32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
      %649 = mhlo.copy %648 : tensor<2x4x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %650 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %651 = stablehlo.compare LT, %649, %650 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
      %c_2 = stablehlo.constant dense<8> : tensor<i32>
      %652 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %653 = stablehlo.add %649, %652 : tensor<2x4x1x1xi32>
      %654 = stablehlo.select %651, %653, %649 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
      %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %657 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %658 = stablehlo.compare LT, %647, %657 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %659 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %660 = stablehlo.add %647, %659 : tensor<2x4xi32>
      %661 = stablehlo.select %658, %660, %647 : tensor<2x4xi1>, tensor<2x4xi32>
      %662 = stablehlo.broadcast_in_dim %661, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %663 = stablehlo.concatenate %656, %662, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %664 = mhlo.bitcast %663 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      mhlo.return %664 : tensor<64x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<64x2xi32>
    %3 = "mhlo.fusion"(%arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512xf32>
      %644 = stablehlo.rsqrt %643 : tensor<512xf32>
      mhlo.return %644 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.fusion"(%arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512xf32>
      %644 = stablehlo.rsqrt %643 : tensor<512xf32>
      mhlo.return %644 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %5 = "mhlo.fusion"(%arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<768xf32>
      %644 = stablehlo.rsqrt %643 : tensor<768xf32>
      mhlo.return %644 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %6 = "mhlo.fusion"(%arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<768xf32>
      %644 = stablehlo.rsqrt %643 : tensor<768xf32>
      mhlo.return %644 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %7 = "mhlo.fusion"(%arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<384xf32>
      %644 = stablehlo.rsqrt %643 : tensor<384xf32>
      mhlo.return %644 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %8 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %642 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<512x192x16x16xf32>
      mhlo.return %642 : tensor<512x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<512x192x16x16xf32>
    %9 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %642 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %643 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %644 = stablehlo.multiply %642, %643 : tensor<8xi32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %646 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %647 = stablehlo.add %645, %646 : tensor<2x8xi32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
      %649 = mhlo.copy %648 : tensor<2x8x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %650 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %651 = stablehlo.compare LT, %649, %650 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
      %c_2 = stablehlo.constant dense<16> : tensor<i32>
      %652 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %653 = stablehlo.add %649, %652 : tensor<2x8x1x1xi32>
      %654 = stablehlo.select %651, %653, %649 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
      %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %657 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %658 = stablehlo.compare LT, %647, %657 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %659 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %660 = stablehlo.add %647, %659 : tensor<2x8xi32>
      %661 = stablehlo.select %658, %660, %647 : tensor<2x8xi1>, tensor<2x8xi32>
      %662 = stablehlo.broadcast_in_dim %661, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %663 = stablehlo.concatenate %656, %662, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %664 = mhlo.bitcast %663 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      mhlo.return %664 : tensor<256x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<256x2xi32>
    %10 = "mhlo.fusion"(%arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<384xf32>
      %644 = stablehlo.rsqrt %643 : tensor<384xf32>
      mhlo.return %644 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %11 = "mhlo.fusion"(%arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<384xf32>
      %644 = stablehlo.rsqrt %643 : tensor<384xf32>
      mhlo.return %644 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %12 = "mhlo.fusion"(%arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512xf32>
      %644 = stablehlo.rsqrt %643 : tensor<512xf32>
      mhlo.return %644 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %13 = "mhlo.fusion"(%arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512xf32>
      %644 = stablehlo.rsqrt %643 : tensor<512xf32>
      mhlo.return %644 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %14 = "mhlo.fusion"(%arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<f32>):
      %642 = stablehlo.broadcast_in_dim %arg245, dims = [] : (tensor<f32>) -> tensor<512x128x32x32xf32>
      mhlo.return %642 : tensor<512x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<f32>) -> tensor<512x128x32x32xf32>
    %16 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %642 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %643 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %644 = stablehlo.add %642, %643 : tensor<2x16xi32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
      %646 = mhlo.copy %645 : tensor<2x16x1x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %647 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %648 = stablehlo.compare LT, %646, %647 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
      %c_1 = stablehlo.constant dense<32> : tensor<i32>
      %649 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %650 = stablehlo.add %646, %649 : tensor<2x16x1x1xi32>
      %651 = stablehlo.select %648, %650, %646 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
      %652 = mhlo.bitcast %651 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
      %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %654 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %655 = stablehlo.compare LT, %644, %654 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %656 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %657 = stablehlo.add %644, %656 : tensor<2x16xi32>
      %658 = stablehlo.select %655, %657, %644 : tensor<2x16xi1>, tensor<2x16xi32>
      %659 = stablehlo.broadcast_in_dim %658, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %660 = stablehlo.concatenate %653, %659, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %661 = mhlo.bitcast %660 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      mhlo.return %661 : tensor<1024x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<1024x2xi32>
    %17 = "mhlo.fusion"(%arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %19 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %20 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %21 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<128xf32>
      %644 = stablehlo.rsqrt %643 : tensor<128xf32>
      mhlo.return %644 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %22 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %23 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<256xf32>
      %644 = stablehlo.rsqrt %643 : tensor<256xf32>
      mhlo.return %644 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %24 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<128xf32>
      %644 = stablehlo.rsqrt %643 : tensor<128xf32>
      mhlo.return %644 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %25 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<128xf32>
      %644 = stablehlo.rsqrt %643 : tensor<128xf32>
      mhlo.return %644 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %26 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<128xf32>
      %644 = stablehlo.rsqrt %643 : tensor<128xf32>
      mhlo.return %644 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %27 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<64xf32>
      %644 = stablehlo.rsqrt %643 : tensor<64xf32>
      mhlo.return %644 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %28 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<64xf32>
      %644 = stablehlo.rsqrt %643 : tensor<64xf32>
      mhlo.return %644 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %29 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<64xf32>
      %644 = stablehlo.rsqrt %643 : tensor<64xf32>
      mhlo.return %644 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %30 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<32xf32>
      %644 = stablehlo.rsqrt %643 : tensor<32xf32>
      mhlo.return %644 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %31 = "mhlo.fusion"(%arg244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x3x256x256xf32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,256,3]{2,1,3,0}"} : (tensor<512x3x256x256xf32>) -> tensor<512x256x256x3xf32>
      %643 = mhlo.copy %642 : tensor<512x256x256x3xf32>
      mhlo.return %643 : tensor<512x256x256x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x3x256x256xf32>) -> tensor<512x256x256x3xf32>
    %32 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32x3x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,3,32]{1,0,2,3}"} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
      %643 = mhlo.copy %642 : tensor<3x3x3x32xf32>
      mhlo.return %643 : tensor<3x3x3x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32x3x3x3xf32>) -> tensor<3x3x3x32xf32>
    %33 = stablehlo.convolution(%31, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x256x256x3xf32>, tensor<3x3x3x32xf32>) -> tensor<512x128x128x32xf32>
    %34 = "mhlo.fusion"(%arg2, %arg1, %30, %33, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<512x128x128x32xf32>, %arg249: tensor<32xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x128x128x32xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x128x128x32xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x128x128x32xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<32xf32>) -> tensor<512x128x128x32xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x128x128x32xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,128,128]{1,3,2,0}"} : (tensor<512x128x128x32xf32>) -> tensor<512x32x128x128xf32>
      %651 = mhlo.copy %650 : tensor<512x32x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x32x128x128xf32>
      %653 = stablehlo.negate %651 : tensor<512x32x128x128xf32>
      %654 = stablehlo.exponential %653 : tensor<512x32x128x128xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x32x128x128xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x32x128x128xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x32x128x128xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,32]{2,1,3,0}"} : (tensor<512x32x128x128xf32>) -> tensor<512x128x128x32xf32>
      %659 = mhlo.copy %658 : tensor<512x128x128x32xf32>
      mhlo.return %659 : tensor<512x128x128x32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<512x128x128x32xf32>, tensor<32xf32>) -> tensor<512x128x128x32xf32>
    %35 = mhlo.bitcast %34 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x128x32xf32>) -> tensor<8388608x32xf32>
    %36 = "mhlo.fusion"(%arg3) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x32x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,32,64]{1,0,2,3}"} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
      %643 = mhlo.copy %642 : tensor<1x1x32x64xf32>
      mhlo.return %643 : tensor<1x1x32x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x32x1x1xf32>) -> tensor<1x1x32x64xf32>
    %37 = mhlo.bitcast %36 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x32x64xf32>) -> tensor<32x64xf32>
    %38 = stablehlo.dot %35, %37, precision = [DEFAULT, DEFAULT] : (tensor<8388608x32xf32>, tensor<32x64xf32>) -> tensor<8388608x64xf32>
    %39 = mhlo.bitcast %38 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8388608x64xf32>) -> tensor<512x128x128x64xf32>
    %40 = "mhlo.fusion"(%arg5, %arg4, %29, %39, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<512x128x128x64xf32>, %arg249: tensor<64xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x128x128x64xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x128x128x64xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x128x128x64xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x128x128x64xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,128,128]{1,3,2,0}"} : (tensor<512x128x128x64xf32>) -> tensor<512x64x128x128xf32>
      %651 = mhlo.copy %650 : tensor<512x64x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x64x128x128xf32>
      %653 = stablehlo.negate %651 : tensor<512x64x128x128xf32>
      %654 = stablehlo.exponential %653 : tensor<512x64x128x128xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x64x128x128xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x64x128x128xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x64x128x128xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,64]{2,1,3,0}"} : (tensor<512x64x128x128xf32>) -> tensor<512x128x128x64xf32>
      %659 = mhlo.copy %658 : tensor<512x128x128x64xf32>
      mhlo.return %659 : tensor<512x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<512x128x128x64xf32>, tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %41 = "mhlo.fusion"(%arg6) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,64]{1,0,2,3}"} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x64xf32>
      mhlo.return %643 : tensor<3x3x1x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1x3x3xf32>) -> tensor<3x3x1x64xf32>
    %42 = stablehlo.convolution(%40, %41) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 64 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x128x128x64xf32>, tensor<3x3x1x64xf32>) -> tensor<512x128x128x64xf32>
    %43 = "mhlo.fusion"(%arg8, %arg7, %28, %42, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<512x128x128x64xf32>, %arg249: tensor<64xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x128x128x64xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x128x128x64xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x128x128x64xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x128x128x64xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,128,128]{1,3,2,0}"} : (tensor<512x128x128x64xf32>) -> tensor<512x64x128x128xf32>
      %651 = mhlo.copy %650 : tensor<512x64x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x64x128x128xf32>
      %653 = stablehlo.negate %651 : tensor<512x64x128x128xf32>
      %654 = stablehlo.exponential %653 : tensor<512x64x128x128xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x64x128x128xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x64x128x128xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x64x128x128xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,64]{2,1,3,0}"} : (tensor<512x64x128x128xf32>) -> tensor<512x128x128x64xf32>
      %659 = mhlo.copy %658 : tensor<512x128x128x64xf32>
      mhlo.return %659 : tensor<512x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<512x128x128x64xf32>, tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %44 = mhlo.bitcast %43 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x128x64xf32>) -> tensor<8388608x64xf32>
    %45 = "mhlo.fusion"(%arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x64x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,64]{1,0,2,3}"} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
      %643 = mhlo.copy %642 : tensor<1x1x64x64xf32>
      mhlo.return %643 : tensor<1x1x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64x1x1xf32>) -> tensor<1x1x64x64xf32>
    %46 = mhlo.bitcast %45 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
    %47 = stablehlo.dot %44, %46, precision = [DEFAULT, DEFAULT] : (tensor<8388608x64xf32>, tensor<64x64xf32>) -> tensor<8388608x64xf32>
    %48 = mhlo.bitcast %47 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8388608x64xf32>) -> tensor<512x128x128x64xf32>
    %49 = "mhlo.fusion"(%arg11, %arg10, %27, %48, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<512x128x128x64xf32>, %arg249: tensor<64xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x128x128x64xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x128x128x64xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x128x128x64xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<64xf32>) -> tensor<512x128x128x64xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x128x128x64xf32>
      mhlo.return %649 : tensor<512x128x128x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<512x128x128x64xf32>, tensor<64xf32>) -> tensor<512x128x128x64xf32>
    %50 = mhlo.bitcast %49 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x128x64xf32>) -> tensor<8388608x64xf32>
    %51 = "mhlo.fusion"(%arg12) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x64x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,64,128]{1,0,2,3}"} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x64x128xf32>
      mhlo.return %643 : tensor<1x1x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x64x1x1xf32>) -> tensor<1x1x64x128xf32>
    %52 = mhlo.bitcast %51 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x64x128xf32>) -> tensor<64x128xf32>
    %53 = stablehlo.dot %50, %52, precision = [DEFAULT, DEFAULT] : (tensor<8388608x64xf32>, tensor<64x128xf32>) -> tensor<8388608x128xf32>
    %54 = mhlo.bitcast %53 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8388608x128xf32>) -> tensor<512x128x128x128xf32>
    %55 = "mhlo.fusion"(%arg14, %arg13, %26, %54, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<512x128x128x128xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x128x128x128xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x128x128x128xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x128x128x128xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<512x128x128x128xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x128x128x128xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,128]{1,3,2,0}"} : (tensor<512x128x128x128xf32>) -> tensor<512x128x128x128xf32>
      %651 = mhlo.copy %650 : tensor<512x128x128x128xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x128x128x128xf32>
      %653 = stablehlo.negate %651 : tensor<512x128x128x128xf32>
      %654 = stablehlo.exponential %653 : tensor<512x128x128x128xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x128x128x128xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x128x128x128xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x128x128x128xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,128,128]{2,1,3,0}"} : (tensor<512x128x128x128xf32>) -> tensor<512x128x128x128xf32>
      %659 = mhlo.copy %658 : tensor<512x128x128x128xf32>
      mhlo.return %659 : tensor<512x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x128x128xf32>, tensor<128xf32>) -> tensor<512x128x128x128xf32>
    %56 = "mhlo.fusion"(%arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,128]{1,0,2,3}"} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x128xf32>
      mhlo.return %643 : tensor<3x3x1x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x1x3x3xf32>) -> tensor<3x3x1x128xf32>
    %57 = stablehlo.convolution(%55, %56) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 128 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x128x128x128xf32>, tensor<3x3x1x128xf32>) -> tensor<512x64x64x128xf32>
    %58 = "mhlo.fusion"(%arg17, %arg16, %25, %57, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<512x64x64x128xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x64x64x128xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x64x64x128xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x64x64x128xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x64x64x128xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,64,64]{1,3,2,0}"} : (tensor<512x64x64x128xf32>) -> tensor<512x128x64x64xf32>
      %651 = mhlo.copy %650 : tensor<512x128x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x128x64x64xf32>
      %653 = stablehlo.negate %651 : tensor<512x128x64x64xf32>
      %654 = stablehlo.exponential %653 : tensor<512x128x64x64xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x128x64x64xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x128x64x64xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x128x64x64xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,128]{2,1,3,0}"} : (tensor<512x128x64x64xf32>) -> tensor<512x64x64x128xf32>
      %659 = mhlo.copy %658 : tensor<512x64x64x128xf32>
      mhlo.return %659 : tensor<512x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x64x64x128xf32>, tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %59 = mhlo.bitcast %58 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x64x64x128xf32>) -> tensor<2097152x128xf32>
    %60 = "mhlo.fusion"(%arg18) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x128xf32>
      mhlo.return %643 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %61 = mhlo.bitcast %60 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %62 = stablehlo.dot %59, %61, precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<128x128xf32>) -> tensor<2097152x128xf32>
    %63 = mhlo.bitcast %62 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x128xf32>) -> tensor<512x64x64x128xf32>
    %64 = "mhlo.fusion"(%arg20, %arg19, %24, %63, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<512x64x64x128xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x64x64x128xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x64x64x128xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x64x64x128xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x64x64x128xf32>
      mhlo.return %649 : tensor<512x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x64x64x128xf32>, tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x64x64x128xf32>) -> tensor<2097152x128xf32>
    %66 = "mhlo.fusion"(%arg21) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x256xf32>
      mhlo.return %643 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %67 = mhlo.bitcast %66 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %68 = stablehlo.dot %65, %67, precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<128x256xf32>) -> tensor<2097152x256xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x256xf32>) -> tensor<512x64x64x256xf32>
    %70 = "mhlo.fusion"(%arg23, %arg22, %23, %69, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x64x64x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x64x64x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x64x64x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x64x64x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x64x64x256xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,64,64]{1,3,2,0}"} : (tensor<512x64x64x256xf32>) -> tensor<512x256x64x64xf32>
      %651 = mhlo.copy %650 : tensor<512x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x64x64xf32>
      %653 = stablehlo.negate %651 : tensor<512x256x64x64xf32>
      %654 = stablehlo.exponential %653 : tensor<512x256x64x64xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x256x64x64xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x256x64x64xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x256x64x64xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,256]{2,1,3,0}"} : (tensor<512x256x64x64xf32>) -> tensor<512x64x64x256xf32>
      %659 = mhlo.copy %658 : tensor<512x64x64x256xf32>
      mhlo.return %659 : tensor<512x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x64x64x256xf32>, tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %71 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x256xf32>
      mhlo.return %643 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %72 = stablehlo.convolution(%70, %71) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<512x64x64x256xf32>
    %73 = "mhlo.fusion"(%arg26, %arg25, %22, %72, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x64x64x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x64x64x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x64x64x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x64x64x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x64x64x256xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,64,64]{1,3,2,0}"} : (tensor<512x64x64x256xf32>) -> tensor<512x256x64x64xf32>
      %651 = mhlo.copy %650 : tensor<512x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x64x64xf32>
      %653 = stablehlo.negate %651 : tensor<512x256x64x64xf32>
      %654 = stablehlo.exponential %653 : tensor<512x256x64x64xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x256x64x64xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x256x64x64xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x256x64x64xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,256]{2,1,3,0}"} : (tensor<512x256x64x64xf32>) -> tensor<512x64x64x256xf32>
      %659 = mhlo.copy %658 : tensor<512x64x64x256xf32>
      mhlo.return %659 : tensor<512x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x64x64x256xf32>, tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %74 = mhlo.bitcast %73 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x64x64x256xf32>) -> tensor<2097152x256xf32>
    %75 = "mhlo.fusion"(%arg27) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x128xf32>
      mhlo.return %643 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %77 = stablehlo.dot %74, %76, precision = [DEFAULT, DEFAULT] : (tensor<2097152x256xf32>, tensor<256x128xf32>) -> tensor<2097152x128xf32>
    %78 = mhlo.bitcast %77 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x128xf32>) -> tensor<512x64x64x128xf32>
    %79 = "mhlo.fusion"(%arg29, %arg28, %21, %78, %arg212, %arg20, %arg19, %24, %63, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<512x64x64x128xf32>, %arg249: tensor<128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>, %arg252: tensor<128xf32>, %arg253: tensor<512x64x64x128xf32>, %arg254: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg254, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %643 = stablehlo.subtract %arg253, %642 : tensor<512x64x64x128xf32>
      %644 = stablehlo.broadcast_in_dim %arg252, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x64x64x128xf32>
      %646 = stablehlo.broadcast_in_dim %arg251, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x64x64x128xf32>
      %648 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x64x64x128xf32>
      %650 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %651 = stablehlo.subtract %arg248, %650 : tensor<512x64x64x128xf32>
      %652 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %653 = stablehlo.multiply %651, %652 : tensor<512x64x64x128xf32>
      %654 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %655 = stablehlo.multiply %653, %654 : tensor<512x64x64x128xf32>
      %656 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<128xf32>) -> tensor<512x64x64x128xf32>
      %657 = stablehlo.add %655, %656 : tensor<512x64x64x128xf32>
      %658 = stablehlo.add %649, %657 : tensor<512x64x64x128xf32>
      mhlo.return %658 : tensor<512x64x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x64x64x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x64x64x128xf32>, tensor<128xf32>) -> tensor<512x64x64x128xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x64x64x128xf32>) -> tensor<2097152x128xf32>
    %81 = "mhlo.fusion"(%arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x256xf32>
      mhlo.return %643 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %82 = mhlo.bitcast %81 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %83 = stablehlo.dot %80, %82, precision = [DEFAULT, DEFAULT] : (tensor<2097152x128xf32>, tensor<128x256xf32>) -> tensor<2097152x256xf32>
    %84 = mhlo.bitcast %83 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2097152x256xf32>) -> tensor<512x64x64x256xf32>
    %85 = "mhlo.fusion"(%arg32, %arg31, %20, %84, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x64x64x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x64x64x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x64x64x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x64x64x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x64x64x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x64x64x256xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,64,64]{1,3,2,0}"} : (tensor<512x64x64x256xf32>) -> tensor<512x256x64x64xf32>
      %651 = mhlo.copy %650 : tensor<512x256x64x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x64x64xf32>
      %653 = stablehlo.negate %651 : tensor<512x256x64x64xf32>
      %654 = stablehlo.exponential %653 : tensor<512x256x64x64xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x256x64x64xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x256x64x64xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x256x64x64xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,64,64,256]{2,1,3,0}"} : (tensor<512x256x64x64xf32>) -> tensor<512x64x64x256xf32>
      %659 = mhlo.copy %658 : tensor<512x64x64x256xf32>
      mhlo.return %659 : tensor<512x64x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x64x64x256xf32>, tensor<256xf32>) -> tensor<512x64x64x256xf32>
    %86 = "mhlo.fusion"(%arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x256xf32>
      mhlo.return %643 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %87 = stablehlo.convolution(%85, %86) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x64x64x256xf32>, tensor<3x3x1x256xf32>) -> tensor<512x32x32x256xf32>
    %88 = "mhlo.fusion"(%arg35, %arg34, %19, %87, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x32x32x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x32x32x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x32x32x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x32x32x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x32x32x256xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,32,32]{1,3,2,0}"} : (tensor<512x32x32x256xf32>) -> tensor<512x256x32x32xf32>
      %651 = mhlo.copy %650 : tensor<512x256x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x32x32xf32>
      %653 = stablehlo.negate %651 : tensor<512x256x32x32xf32>
      %654 = stablehlo.exponential %653 : tensor<512x256x32x32xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x256x32x32xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x256x32x32xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x256x32x32xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,256]{2,1,3,0}"} : (tensor<512x256x32x32xf32>) -> tensor<512x32x32x256xf32>
      %659 = mhlo.copy %658 : tensor<512x32x32x256xf32>
      mhlo.return %659 : tensor<512x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x32x32x256xf32>, tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %89 = mhlo.bitcast %88 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x32x32x256xf32>) -> tensor<524288x256xf32>
    %90 = "mhlo.fusion"(%arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x256xf32>
      mhlo.return %643 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %91 = mhlo.bitcast %90 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %92 = stablehlo.dot %89, %91, precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<256x256xf32>) -> tensor<524288x256xf32>
    %93 = mhlo.bitcast %92 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<512x32x32x256xf32>
    %94 = "mhlo.fusion"(%arg38, %arg37, %18, %93, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x32x32x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x32x32x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x32x32x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x32x32x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x32x32x256xf32>
      mhlo.return %649 : tensor<512x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x32x32x256xf32>, tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %95 = "mhlo.fusion"(%arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,256]{1,0,2,3}"} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x256xf32>
      mhlo.return %643 : tensor<3x3x1x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1x3x3xf32>) -> tensor<3x3x1x256xf32>
    %96 = stablehlo.convolution(%94, %95) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 256 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x32x32x256xf32>, tensor<3x3x1x256xf32>) -> tensor<512x32x32x256xf32>
    %97 = "mhlo.fusion"(%arg41, %arg40, %17, %96, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x32x32x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x32x32x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x32x32x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x32x32x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x32x32x256xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,32,32]{1,3,2,0}"} : (tensor<512x32x32x256xf32>) -> tensor<512x256x32x32xf32>
      %651 = mhlo.copy %650 : tensor<512x256x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x32x32xf32>
      %653 = stablehlo.negate %651 : tensor<512x256x32x32xf32>
      %654 = stablehlo.exponential %653 : tensor<512x256x32x32xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x256x32x32xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x256x32x32xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x256x32x32xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,256]{2,1,3,0}"} : (tensor<512x256x32x32xf32>) -> tensor<512x32x32x256xf32>
      %659 = mhlo.copy %658 : tensor<512x32x32x256xf32>
      mhlo.return %659 : tensor<512x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x32x32x256xf32>, tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %98 = mhlo.bitcast %97 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x32x32x256xf32>) -> tensor<524288x256xf32>
    %99 = "mhlo.fusion"(%arg42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x128xf32>
      mhlo.return %643 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %100 = mhlo.bitcast %99 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %101 = stablehlo.dot %98, %100, precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<256x128xf32>) -> tensor<524288x128xf32>
    %102 = mhlo.bitcast %101 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x128xf32>) -> tensor<512x32x32x128xf32>
    %103 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %642 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %643 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %644 = stablehlo.add %642, %643 : tensor<2x16xi32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,16,1,1]{1,0,3,2}"} : (tensor<2x16xi32>) -> tensor<2x16x1x1xi32>
      %646 = mhlo.copy %645 : tensor<2x16x1x1xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %647 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %648 = stablehlo.compare LT, %646, %647 : (tensor<2x16x1x1xi32>, tensor<2x16x1x1xi32>) -> tensor<2x16x1x1xi1>
      %c_1 = stablehlo.constant dense<32> : tensor<i32>
      %649 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16x1x1xi32>
      %650 = stablehlo.add %646, %649 : tensor<2x16x1x1xi32>
      %651 = stablehlo.select %648, %650, %646 : tensor<2x16x1x1xi1>, tensor<2x16x1x1xi32>
      %652 = mhlo.bitcast %651 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x16x1x1xi32>) -> tensor<2x16xi32>
      %653 = stablehlo.broadcast_in_dim %652, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %654 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %655 = stablehlo.compare LT, %644, %654 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %656 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %657 = stablehlo.add %644, %656 : tensor<2x16xi32>
      %658 = stablehlo.select %655, %657, %644 : tensor<2x16xi1>, tensor<2x16xi32>
      %659 = stablehlo.broadcast_in_dim %658, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %660 = stablehlo.concatenate %653, %659, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      mhlo.return %660 : tensor<2x16x2x16x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x16x2x16x2xi32>
    %104 = "mhlo.fusion"(%102, %103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x32x32x128xf32>, %arg246: tensor<2x16x2x16x2xi32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,32,32]{1,3,2,0}"} : (tensor<512x32x32x128xf32>) -> tensor<512x128x32x32xf32>
      %643 = mhlo.copy %642 : tensor<512x128x32x32xf32>
      %644 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %645 = "stablehlo.gather"(%643, %644) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 512, 128, 1, 1>}> : (tensor<512x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x512x128x1x1xf32>
      mhlo.return %645 : tensor<1024x512x128x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x32x32x128xf32>, tensor<2x16x2x16x2xi32>) -> tensor<1024x512x128x1x1xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x512x128x1x1xf32>):
      %642 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
      %643 = stablehlo.transpose %642, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
      %644 = mhlo.copy %643 : tensor<512x128x2x2x16x16xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<1x512x131072xf32>
      mhlo.return %645 : tensor<1x512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x512x128x1x1xf32>) -> tensor<1x512x131072xf32>
    %106 = "mhlo.fusion"(%105, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
      mhlo.return %642 : tensor<1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x512x128xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    %108 = "mhlo.fusion"(%107, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<1024x512x128x1x1xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
      %643 = stablehlo.transpose %642, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
      %644 = mhlo.copy %643 : tensor<512x128x2x2x16x16xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %647 = stablehlo.multiply %arg245, %646 : tensor<512xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %649 = stablehlo.subtract %645, %648 : tensor<512x131072xf32>
      %650 = stablehlo.multiply %649, %649 : tensor<512x131072xf32>
      mhlo.return %650 : tensor<512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<1024x512x128x1x1xf32>) -> tensor<512x131072xf32>
    %109 = "mhlo.fusion"(%108, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
      mhlo.return %642 : tensor<512x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %110 = "mhlo.fusion"(%109, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x4096xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
      mhlo.return %642 : tensor<512x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4xf32>) -> tensor<512xf32>
    %112 = "mhlo.fusion"(%arg43, %111, %107, %104, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1024x512x128x1x1xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
      %644 = stablehlo.transpose %643, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
      %645 = mhlo.copy %644 : tensor<512x128x2x2x16x16xf32>
      %646 = mhlo.bitcast %645 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %648 = stablehlo.multiply %arg247, %647 : tensor<512xf32>
      %649 = stablehlo.broadcast_in_dim %648, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %650 = stablehlo.subtract %646, %649 : tensor<512x131072xf32>
      %651 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x131072xf32>
      %653 = mhlo.bitcast %652 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
      %654 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %655 = stablehlo.multiply %653, %654 : tensor<512x128x4x256xf32>
      %656 = stablehlo.add %642, %655 : tensor<512x128x4x256xf32>
      %657 = stablehlo.transpose %656, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
      %658 = mhlo.copy %657 : tensor<512x4x256x128xf32>
      mhlo.return %658 : tensor<512x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1024x512x128x1x1xf32>, tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %113 = mhlo.bitcast %112 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x128xf32>) -> tensor<524288x128xf32>
    %114 = "mhlo.fusion"(%arg45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<257x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x257xf32>
      mhlo.return %643 : tensor<1x1x128x257xf32>
    }) {output_operand_aliasing = []} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %115 = mhlo.bitcast %114 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x257xf32>) -> tensor<128x257xf32>
    %116 = stablehlo.dot %113, %115, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x257xf32>) -> tensor<524288x257xf32>
    %117 = mhlo.bitcast %116 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x257xf32>) -> tensor<512x4x256x257xf32>
    %118 = "mhlo.fusion"(%117, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x256x257xf32>, %arg246: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
      mhlo.return %646 : tensor<512x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x1x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %119 = "mhlo.fusion"(%118, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4x8xf32>
      mhlo.return %642 : tensor<512x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4x8xf32>
    %120 = "mhlo.fusion"(%119, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x8xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x8xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x8xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %121 = "mhlo.fusion"(%120, %117, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x256x257xf32>, %arg247: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x256xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x256xf32>
      mhlo.return %652 : tensor<512x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x1x4x256xf32>
    %122 = "mhlo.fusion"(%121, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %123 = "mhlo.fusion"(%121, %122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x256xf32>
      mhlo.return %644 : tensor<512x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x256xf32>
    %124 = "mhlo.fusion"(%123, %117, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<512x4x256x257xf32>, %arg247: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 1:129, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x256xf32>) -> tensor<512x4x256xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x256xf32>) -> tensor<512x128x4x256xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x128x4x256xf32>
      mhlo.return %649 : tensor<512x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x128x4x256xf32>
    %125 = "mhlo.fusion"(%124, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4x256xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x128x4x256xf32>, tensor<f32>) -> tensor<512x128x4x8xf32>
      mhlo.return %642 : tensor<512x128x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x256xf32>, tensor<f32>) -> tensor<512x128x4x8xf32>
    %126 = "mhlo.fusion"(%125, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x128x4x8xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x128x4x8xf32>, tensor<f32>) -> tensor<512x128x4xf32>
      mhlo.return %642 : tensor<512x128x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x8xf32>, tensor<f32>) -> tensor<512x128x4xf32>
    %127 = "mhlo.fusion"(%126, %117, %arg46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4xf32>, %arg246: tensor<512x4x256x257xf32>, %arg247: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 129:257, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x128x4x256xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x128x4x256xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x128x4xf32>) -> tensor<512x128x4x256xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x128x4x256xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
      %652 = mhlo.copy %651 : tensor<512x4x256x128xf32>
      mhlo.return %652 : tensor<512x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4xf32>, tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x4x256x128xf32>
    %128 = mhlo.bitcast %127 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x128xf32>) -> tensor<524288x128xf32>
    %129 = "mhlo.fusion"(%arg47) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x128xf32>
      mhlo.return %643 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %130 = mhlo.bitcast %129 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %131 = stablehlo.dot %128, %130, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x128xf32>) -> tensor<524288x128xf32>
    %132 = mhlo.bitcast %131 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x128xf32>) -> tensor<512x4x256x128xf32>
    %133 = "mhlo.fusion"(%132, %arg48, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x256x128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<1024x512x128x1x1xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x256x128xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x128x4x256xf32>
      %646 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x512x128x1x1xf32>) -> tensor<2x16x2x16x512x128xf32>
      %647 = stablehlo.transpose %646, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,128,2,2,16,16]{1,0,5,3,4,2}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<512x128x2x2x16x16xf32>
      %648 = mhlo.copy %647 : tensor<512x128x2x2x16x16xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x128x2x2x16x16xf32>) -> tensor<512x128x4x256xf32>
      %650 = stablehlo.add %645, %649 : tensor<512x128x4x256xf32>
      mhlo.return %650 : tensor<512x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x256x128xf32>, tensor<128xf32>, tensor<1024x512x128x1x1xf32>) -> tensor<512x128x4x256xf32>
    %134 = mhlo.bitcast %133 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<1x512x131072xf32>
    %135 = "mhlo.fusion"(%134, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
      mhlo.return %642 : tensor<1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    %136 = "mhlo.fusion"(%135, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x512x128xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    %137 = "mhlo.fusion"(%136, %133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x128x4x256xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x131072xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x131072xf32>
      mhlo.return %647 : tensor<512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %138 = "mhlo.fusion"(%137, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
      mhlo.return %642 : tensor<512x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %139 = "mhlo.fusion"(%138, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x4096xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
      mhlo.return %642 : tensor<512x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    %140 = "mhlo.fusion"(%139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4xf32>) -> tensor<512xf32>
    %141 = "mhlo.fusion"(%arg49, %140, %136, %133, %arg50) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x128x4x256xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x131072xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x131072xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x128x4x256xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x128x4x256xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
      %655 = mhlo.copy %654 : tensor<512x4x256x128xf32>
      mhlo.return %655 : tensor<512x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x4x256xf32>, tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %142 = mhlo.bitcast %141 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x128xf32>) -> tensor<524288x128xf32>
    %143 = "mhlo.fusion"(%arg51) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x256xf32>
      mhlo.return %643 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %144 = mhlo.bitcast %143 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %145 = stablehlo.dot %142, %144, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x256xf32>) -> tensor<524288x256xf32>
    %146 = mhlo.bitcast %145 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<512x4x256x256xf32>
    %147 = "mhlo.fusion"(%146, %arg52) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x256x256xf32>, %arg246: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x4x256x256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x256x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,256]{1,3,2,0}"} : (tensor<512x4x256x256xf32>) -> tensor<512x256x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x4x256xf32>
      %647 = stablehlo.negate %645 : tensor<512x256x4x256xf32>
      %648 = stablehlo.exponential %647 : tensor<512x256x4x256xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x256x4x256xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x256x4x256xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x256x4x256xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,256]{2,1,3,0}"} : (tensor<512x256x4x256xf32>) -> tensor<512x4x256x256xf32>
      %653 = mhlo.copy %652 : tensor<512x4x256x256xf32>
      mhlo.return %653 : tensor<512x4x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x256x256xf32>, tensor<256xf32>) -> tensor<512x4x256x256xf32>
    %148 = mhlo.bitcast %147 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x256xf32>) -> tensor<524288x256xf32>
    %149 = "mhlo.fusion"(%arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x128xf32>
      mhlo.return %643 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %150 = mhlo.bitcast %149 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %151 = stablehlo.dot %148, %150, precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<256x128xf32>) -> tensor<524288x128xf32>
    %152 = mhlo.bitcast %151 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x128xf32>) -> tensor<512x4x256x128xf32>
    %153 = "mhlo.fusion"(%133, %152, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4x256xf32>, %arg246: tensor<512x4x256x128xf32>, %arg247: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x128xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x128x4x256xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x128x4x256xf32>
      mhlo.return %646 : tensor<512x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x256xf32>, tensor<512x4x256x128xf32>, tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %154 = mhlo.bitcast %153 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<1x512x131072xf32>
    %155 = "mhlo.fusion"(%154, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
      mhlo.return %642 : tensor<1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    %156 = "mhlo.fusion"(%155, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x512x128xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    %157 = "mhlo.fusion"(%156, %153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x128x4x256xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x131072xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x131072xf32>
      mhlo.return %647 : tensor<512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %158 = "mhlo.fusion"(%157, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
      mhlo.return %642 : tensor<512x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %159 = "mhlo.fusion"(%158, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x4096xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
      mhlo.return %642 : tensor<512x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    %160 = "mhlo.fusion"(%159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4xf32>) -> tensor<512xf32>
    %161 = "mhlo.fusion"(%arg55, %160, %156, %153, %arg56) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x128x4x256xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x131072xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x131072xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x128x4x256xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x128x4x256xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
      %655 = mhlo.copy %654 : tensor<512x4x256x128xf32>
      mhlo.return %655 : tensor<512x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x4x256xf32>, tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %162 = mhlo.bitcast %161 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x128xf32>) -> tensor<524288x128xf32>
    %163 = "mhlo.fusion"(%arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<257x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,257]{1,0,2,3}"} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x257xf32>
      mhlo.return %643 : tensor<1x1x128x257xf32>
    }) {output_operand_aliasing = []} : (tensor<257x128x1x1xf32>) -> tensor<1x1x128x257xf32>
    %164 = mhlo.bitcast %163 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x257xf32>) -> tensor<128x257xf32>
    %165 = stablehlo.dot %162, %164, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x257xf32>) -> tensor<524288x257xf32>
    %166 = mhlo.bitcast %165 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x257xf32>) -> tensor<512x4x256x257xf32>
    %167 = "mhlo.fusion"(%166, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x256x257xf32>, %arg246: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
      mhlo.return %646 : tensor<512x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x1x4x256xf32>
    %168 = "mhlo.fusion"(%167, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4x8xf32>
      mhlo.return %642 : tensor<512x1x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4x8xf32>
    %169 = "mhlo.fusion"(%168, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x8xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x8xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x8xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %170 = "mhlo.fusion"(%169, %166, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x256x257xf32>, %arg247: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x1x4x256xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x256xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x256xf32>
      mhlo.return %652 : tensor<512x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x1x4x256xf32>
    %171 = "mhlo.fusion"(%170, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %172 = "mhlo.fusion"(%170, %171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x256xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x256xf32>
      mhlo.return %644 : tensor<512x1x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x256xf32>
    %173 = "mhlo.fusion"(%172, %166, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x256xf32>, %arg246: tensor<512x4x256x257xf32>, %arg247: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 1:129, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x256xf32>) -> tensor<512x4x256xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x256xf32>) -> tensor<512x128x4x256xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x128x4x256xf32>
      mhlo.return %649 : tensor<512x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x256xf32>, tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x128x4x256xf32>
    %174 = "mhlo.fusion"(%173, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4x256xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x128x4x256xf32>, tensor<f32>) -> tensor<512x128x4x8xf32>
      mhlo.return %642 : tensor<512x128x4x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x256xf32>, tensor<f32>) -> tensor<512x128x4x8xf32>
    %175 = "mhlo.fusion"(%174, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x128x4x8xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x128x4x8xf32>, tensor<f32>) -> tensor<512x128x4xf32>
      mhlo.return %642 : tensor<512x128x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x8xf32>, tensor<f32>) -> tensor<512x128x4xf32>
    %176 = "mhlo.fusion"(%175, %166, %arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4xf32>, %arg246: tensor<512x4x256x257xf32>, %arg247: tensor<257xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<257xf32>) -> tensor<512x4x256x257xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x257xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,257,4,256]{1,3,2,0}"} : (tensor<512x4x256x257xf32>) -> tensor<512x257x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x257x4x256xf32>
      %646 = stablehlo.slice %645 [0:512, 129:257, 0:4, 0:256] : (tensor<512x257x4x256xf32>) -> tensor<512x128x4x256xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x128x4x256xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x128x4x256xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x128x4xf32>) -> tensor<512x128x4x256xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x128x4x256xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
      %652 = mhlo.copy %651 : tensor<512x4x256x128xf32>
      mhlo.return %652 : tensor<512x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4xf32>, tensor<512x4x256x257xf32>, tensor<257xf32>) -> tensor<512x4x256x128xf32>
    %177 = mhlo.bitcast %176 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x128xf32>) -> tensor<524288x128xf32>
    %178 = "mhlo.fusion"(%arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,128]{1,0,2,3}"} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x128xf32>
      mhlo.return %643 : tensor<1x1x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x128x1x1xf32>) -> tensor<1x1x128x128xf32>
    %179 = mhlo.bitcast %178 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x128xf32>) -> tensor<128x128xf32>
    %180 = stablehlo.dot %177, %179, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x128xf32>) -> tensor<524288x128xf32>
    %181 = mhlo.bitcast %180 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x128xf32>) -> tensor<512x4x256x128xf32>
    %182 = "mhlo.fusion"(%153, %181, %arg60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4x256xf32>, %arg246: tensor<512x4x256x128xf32>, %arg247: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x128xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x128x4x256xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x128x4x256xf32>
      mhlo.return %646 : tensor<512x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x256xf32>, tensor<512x4x256x128xf32>, tensor<128xf32>) -> tensor<512x128x4x256xf32>
    %183 = mhlo.bitcast %182 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<1x512x131072xf32>
    %184 = "mhlo.fusion"(%183, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
      mhlo.return %642 : tensor<1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    %185 = "mhlo.fusion"(%184, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x512x128xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    %186 = "mhlo.fusion"(%185, %182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x128x4x256xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x131072xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x131072xf32>
      mhlo.return %647 : tensor<512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
    %187 = "mhlo.fusion"(%186, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
      mhlo.return %642 : tensor<512x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %188 = "mhlo.fusion"(%187, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x4096xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
      mhlo.return %642 : tensor<512x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4xf32>) -> tensor<512xf32>
    %190 = "mhlo.fusion"(%arg61, %189, %185, %182, %arg62) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x128x4x256xf32>, %arg249: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x131072xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x131072xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x128x4x256xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x128x4x256xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,128]{2,1,3,0}"} : (tensor<512x128x4x256xf32>) -> tensor<512x4x256x128xf32>
      %655 = mhlo.copy %654 : tensor<512x4x256x128xf32>
      mhlo.return %655 : tensor<512x4x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x4x256xf32>, tensor<128xf32>) -> tensor<512x4x256x128xf32>
    %191 = mhlo.bitcast %190 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x128xf32>) -> tensor<524288x128xf32>
    %192 = "mhlo.fusion"(%arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x256xf32>
      mhlo.return %643 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %193 = mhlo.bitcast %192 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %194 = stablehlo.dot %191, %193, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x256xf32>) -> tensor<524288x256xf32>
    %195 = mhlo.bitcast %194 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<512x4x256x256xf32>
    %196 = "mhlo.fusion"(%195, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x256x256xf32>, %arg246: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x4x256x256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x256x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,256]{1,3,2,0}"} : (tensor<512x4x256x256xf32>) -> tensor<512x256x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x256xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x4x256xf32>
      %647 = stablehlo.negate %645 : tensor<512x256x4x256xf32>
      %648 = stablehlo.exponential %647 : tensor<512x256x4x256xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x256x4x256xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x256x4x256xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x256x4x256xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,256,256]{2,1,3,0}"} : (tensor<512x256x4x256xf32>) -> tensor<512x4x256x256xf32>
      %653 = mhlo.copy %652 : tensor<512x4x256x256xf32>
      mhlo.return %653 : tensor<512x4x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x256x256xf32>, tensor<256xf32>) -> tensor<512x4x256x256xf32>
    %197 = mhlo.bitcast %196 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x256x256xf32>) -> tensor<524288x256xf32>
    %198 = "mhlo.fusion"(%arg65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,128]{1,0,2,3}"} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x128xf32>
      mhlo.return %643 : tensor<1x1x256x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128x256x1x1xf32>) -> tensor<1x1x256x128xf32>
    %199 = mhlo.bitcast %198 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x128xf32>) -> tensor<256x128xf32>
    %200 = stablehlo.dot %197, %199, precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<256x128xf32>) -> tensor<524288x128xf32>
    %201 = mhlo.bitcast %200 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x128xf32>) -> tensor<512x4x256x128xf32>
    %202 = "mhlo.fusion"(%182, %201, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x4x256xf32>, %arg246: tensor<512x4x256x128xf32>, %arg247: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x256x128xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x128x4x256xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x128x4x256xf32>
      %647 = mhlo.bitcast %646 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<1x512x131072xf32>
      mhlo.return %647 : tensor<1x512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x4x256xf32>, tensor<512x4x256x128xf32>, tensor<128xf32>) -> tensor<1x512x131072xf32>
    %203 = "mhlo.fusion"(%202, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
      mhlo.return %642 : tensor<1x512x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x131072xf32>, tensor<f32>) -> tensor<1x512x128xf32>
    %204 = "mhlo.fusion"(%203, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1x512x128xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x128xf32>, tensor<f32>) -> tensor<512xf32>
    %205 = "mhlo.fusion"(%204, %182, %201, %arg66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x128x4x256xf32>, %arg247: tensor<512x4x256x128xf32>, %arg248: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
      %643 = stablehlo.add %arg247, %642 : tensor<512x4x256x128xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
      %645 = mhlo.copy %644 : tensor<512x128x4x256xf32>
      %646 = stablehlo.add %645, %arg246 : tensor<512x128x4x256xf32>
      %647 = mhlo.bitcast %646 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %648 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %649 = stablehlo.multiply %arg245, %648 : tensor<512xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %651 = stablehlo.subtract %647, %650 : tensor<512x131072xf32>
      %652 = stablehlo.multiply %651, %651 : tensor<512x131072xf32>
      mhlo.return %652 : tensor<512x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x128x4x256xf32>, tensor<512x4x256x128xf32>, tensor<128xf32>) -> tensor<512x131072xf32>
    %206 = "mhlo.fusion"(%205, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x131072xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
      mhlo.return %642 : tensor<512x4096xf32>
    }) {output_operand_aliasing = []} : (tensor<512x131072xf32>, tensor<f32>) -> tensor<512x4096xf32>
    %207 = "mhlo.fusion"(%206, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x4096xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
      mhlo.return %642 : tensor<512x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4096xf32>, tensor<f32>) -> tensor<512x4xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x4xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4xf32>) -> tensor<512xf32>
    %209 = "mhlo.fusion"(%arg67, %208, %204, %182, %201, %arg66, %arg68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x128x4x256xf32>, %arg249: tensor<512x4x256x128xf32>, %arg250: tensor<128xf32>, %arg251: tensor<128xf32>):
      %642 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %643 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<128xf32>) -> tensor<512x4x256x128xf32>
      %644 = stablehlo.add %arg249, %643 : tensor<512x4x256x128xf32>
      %645 = stablehlo.transpose %644, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,128,4,256]{1,3,2,0}"} : (tensor<512x4x256x128xf32>) -> tensor<512x128x4x256xf32>
      %646 = mhlo.copy %645 : tensor<512x128x4x256xf32>
      %647 = stablehlo.add %646, %arg248 : tensor<512x128x4x256xf32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x131072xf32>
      %cst_1 = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %649 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %650 = stablehlo.multiply %arg247, %649 : tensor<512xf32>
      %651 = stablehlo.broadcast_in_dim %650, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %652 = stablehlo.subtract %648, %651 : tensor<512x131072xf32>
      %653 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x131072xf32>
      %654 = stablehlo.multiply %652, %653 : tensor<512x131072xf32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x131072xf32>) -> tensor<512x128x4x256xf32>
      %656 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<512x128x4x256xf32>
      %657 = stablehlo.multiply %655, %656 : tensor<512x128x4x256xf32>
      %658 = stablehlo.add %642, %657 : tensor<512x128x4x256xf32>
      %659 = mhlo.bitcast %658 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x128x4x256xf32>) -> tensor<512x128x2x2x16x16xf32>
      %660 = stablehlo.transpose %659, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,16,2,16,512,128]{3,1,2,0,5,4}"} : (tensor<512x128x2x2x16x16xf32>) -> tensor<2x16x2x16x512x128xf32>
      %661 = mhlo.copy %660 : tensor<2x16x2x16x512x128xf32>
      %662 = mhlo.bitcast %661 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[1024,512,128,1,1]{2,1,0,4,3}"} : (tensor<2x16x2x16x512x128xf32>) -> tensor<1024x512x128x1x1xf32>
      %663 = mhlo.copy %662 : tensor<1024x512x128x1x1xf32>
      mhlo.return %663 : tensor<1024x512x128x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x128x4x256xf32>, tensor<512x4x256x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1024x512x128x1x1xf32>
    %210 = "mhlo.fusion"(%15, %16, %209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x32x32xf32>, %arg246: tensor<1024x2xi32>, %arg247: tensor<1024x512x128x1x1xf32>):
      %642 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %643 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x512x128x1x1xf32>) -> tensor<512x128x32x32xf32>
      mhlo.return %642 : tensor<512x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x512x128x1x1xf32>) -> tensor<512x128x32x32xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x128x32x32xf32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,128]{2,1,3,0}"} : (tensor<512x128x32x32xf32>) -> tensor<512x32x32x128xf32>
      %643 = mhlo.copy %642 : tensor<512x32x32x128xf32>
      mhlo.return %643 : tensor<512x32x32x128xf32>
    }) {output_operand_aliasing = []} : (tensor<512x128x32x32xf32>) -> tensor<512x32x32x128xf32>
    %212 = mhlo.bitcast %211 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x32x32x128xf32>) -> tensor<524288x128xf32>
    %213 = "mhlo.fusion"(%arg69) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x128x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,128,256]{1,0,2,3}"} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x128x256xf32>
      mhlo.return %643 : tensor<1x1x128x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x128x1x1xf32>) -> tensor<1x1x128x256xf32>
    %214 = mhlo.bitcast %213 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x128x256xf32>) -> tensor<128x256xf32>
    %215 = stablehlo.dot %212, %214, precision = [DEFAULT, DEFAULT] : (tensor<524288x128xf32>, tensor<128x256xf32>) -> tensor<524288x256xf32>
    %216 = mhlo.bitcast %215 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x256xf32>) -> tensor<512x32x32x256xf32>
    %217 = "mhlo.fusion"(%arg71, %arg70, %14, %216, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<512x32x32x256xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x32x32x256xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x32x32x256xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x32x32x256xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<256xf32>) -> tensor<512x32x32x256xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x32x32x256xf32>
      mhlo.return %649 : tensor<512x32x32x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x32x32x256xf32>, tensor<256xf32>) -> tensor<512x32x32x256xf32>
    %218 = mhlo.bitcast %217 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x32x32x256xf32>) -> tensor<524288x256xf32>
    %219 = "mhlo.fusion"(%arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x512xf32>
      mhlo.return %643 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %220 = mhlo.bitcast %219 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %221 = stablehlo.dot %218, %220, precision = [DEFAULT, DEFAULT] : (tensor<524288x256xf32>, tensor<256x512xf32>) -> tensor<524288x512xf32>
    %222 = mhlo.bitcast %221 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<524288x512xf32>) -> tensor<512x32x32x512xf32>
    %223 = "mhlo.fusion"(%arg74, %arg73, %13, %222, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x32x32x512xf32>, %arg249: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x32x32x512xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x32x32x512xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x32x32x512xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<512x32x32x512xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x32x32x512xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,32,32]{1,3,2,0}"} : (tensor<512x32x32x512xf32>) -> tensor<512x512x32x32xf32>
      %651 = mhlo.copy %650 : tensor<512x512x32x32xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512x32x32xf32>
      %653 = stablehlo.negate %651 : tensor<512x512x32x32xf32>
      %654 = stablehlo.exponential %653 : tensor<512x512x32x32xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x512x32x32xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x512x32x32xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x512x32x32xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,32,32,512]{2,1,3,0}"} : (tensor<512x512x32x32xf32>) -> tensor<512x32x32x512xf32>
      %659 = mhlo.copy %658 : tensor<512x32x32x512xf32>
      mhlo.return %659 : tensor<512x32x32x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x32x32x512xf32>, tensor<512xf32>) -> tensor<512x32x32x512xf32>
    %224 = "mhlo.fusion"(%arg75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x512xf32>
      mhlo.return %643 : tensor<3x3x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %225 = stablehlo.convolution(%223, %224) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x32x32x512xf32>, tensor<3x3x1x512xf32>) -> tensor<512x16x16x512xf32>
    %226 = "mhlo.fusion"(%arg77, %arg76, %12, %225, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x16x16x512xf32>, %arg249: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x16x16x512xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x16x16x512xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x16x16x512xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<512x16x16x512xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x16x16x512xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,16,16]{1,3,2,0}"} : (tensor<512x16x16x512xf32>) -> tensor<512x512x16x16xf32>
      %651 = mhlo.copy %650 : tensor<512x512x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512x16x16xf32>
      %653 = stablehlo.negate %651 : tensor<512x512x16x16xf32>
      %654 = stablehlo.exponential %653 : tensor<512x512x16x16xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x512x16x16xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x512x16x16xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x512x16x16xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,512]{2,1,3,0}"} : (tensor<512x512x16x16xf32>) -> tensor<512x16x16x512xf32>
      %659 = mhlo.copy %658 : tensor<512x16x16x512xf32>
      mhlo.return %659 : tensor<512x16x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16x16x512xf32>, tensor<512xf32>) -> tensor<512x16x16x512xf32>
    %227 = mhlo.bitcast %226 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x16x16x512xf32>) -> tensor<131072x512xf32>
    %228 = "mhlo.fusion"(%arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x512x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,384]{1,0,2,3}"} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
      %643 = mhlo.copy %642 : tensor<1x1x512x384xf32>
      mhlo.return %643 : tensor<1x1x512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x512x1x1xf32>) -> tensor<1x1x512x384xf32>
    %229 = mhlo.bitcast %228 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x384xf32>) -> tensor<512x384xf32>
    %230 = stablehlo.dot %227, %229, precision = [DEFAULT, DEFAULT] : (tensor<131072x512xf32>, tensor<512x384xf32>) -> tensor<131072x384xf32>
    %231 = mhlo.bitcast %230 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x384xf32>) -> tensor<512x16x16x384xf32>
    %232 = "mhlo.fusion"(%arg80, %arg79, %11, %231, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<512x16x16x384xf32>, %arg249: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x16x16x384xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x16x16x384xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x16x16x384xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x16x16x384xf32>
      mhlo.return %649 : tensor<512x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<512x16x16x384xf32>, tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %233 = "mhlo.fusion"(%arg81) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,384]{1,0,2,3}"} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x384xf32>
      mhlo.return %643 : tensor<3x3x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x3x3xf32>) -> tensor<3x3x1x384xf32>
    %234 = stablehlo.convolution(%232, %233) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x16x16x384xf32>, tensor<3x3x1x384xf32>) -> tensor<512x16x16x384xf32>
    %235 = "mhlo.fusion"(%arg83, %arg82, %10, %234, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<512x16x16x384xf32>, %arg249: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x16x16x384xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x16x16x384xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x16x16x384xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x16x16x384xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,16,16]{1,3,2,0}"} : (tensor<512x16x16x384xf32>) -> tensor<512x384x16x16xf32>
      %651 = mhlo.copy %650 : tensor<512x384x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x384x16x16xf32>
      %653 = stablehlo.negate %651 : tensor<512x384x16x16xf32>
      %654 = stablehlo.exponential %653 : tensor<512x384x16x16xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x384x16x16xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x384x16x16xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x384x16x16xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,384]{2,1,3,0}"} : (tensor<512x384x16x16xf32>) -> tensor<512x16x16x384xf32>
      %659 = mhlo.copy %658 : tensor<512x16x16x384xf32>
      mhlo.return %659 : tensor<512x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<512x16x16x384xf32>, tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %236 = mhlo.bitcast %235 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x16x16x384xf32>) -> tensor<131072x384xf32>
    %237 = "mhlo.fusion"(%arg84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x384x192xf32>
      mhlo.return %643 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %238 = mhlo.bitcast %237 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %239 = stablehlo.dot %236, %238, precision = [DEFAULT, DEFAULT] : (tensor<131072x384xf32>, tensor<384x192xf32>) -> tensor<131072x192xf32>
    %240 = mhlo.bitcast %239 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x16x16x192xf32>
    %241 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %642 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %643 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %644 = stablehlo.multiply %642, %643 : tensor<8xi32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %646 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %647 = stablehlo.add %645, %646 : tensor<2x8xi32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,8,1,1]{1,0,3,2}"} : (tensor<2x8xi32>) -> tensor<2x8x1x1xi32>
      %649 = mhlo.copy %648 : tensor<2x8x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %650 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %651 = stablehlo.compare LT, %649, %650 : (tensor<2x8x1x1xi32>, tensor<2x8x1x1xi32>) -> tensor<2x8x1x1xi1>
      %c_2 = stablehlo.constant dense<16> : tensor<i32>
      %652 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8x1x1xi32>
      %653 = stablehlo.add %649, %652 : tensor<2x8x1x1xi32>
      %654 = stablehlo.select %651, %653, %649 : tensor<2x8x1x1xi1>, tensor<2x8x1x1xi32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x8x1x1xi32>) -> tensor<2x8xi32>
      %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %657 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %658 = stablehlo.compare LT, %647, %657 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %659 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %660 = stablehlo.add %647, %659 : tensor<2x8xi32>
      %661 = stablehlo.select %658, %660, %647 : tensor<2x8xi1>, tensor<2x8xi32>
      %662 = stablehlo.broadcast_in_dim %661, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %663 = stablehlo.concatenate %656, %662, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      mhlo.return %663 : tensor<2x8x2x8x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x8x2x8x2xi32>
    %242 = "mhlo.fusion"(%240, %241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16x16x192xf32>, %arg246: tensor<2x8x2x8x2xi32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,16,16]{1,3,2,0}"} : (tensor<512x16x16x192xf32>) -> tensor<512x192x16x16xf32>
      %643 = mhlo.copy %642 : tensor<512x192x16x16xf32>
      %644 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %645 = "stablehlo.gather"(%643, %644) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 512, 192, 1, 1>}> : (tensor<512x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x512x192x1x1xf32>
      mhlo.return %645 : tensor<256x512x192x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16x16x192xf32>, tensor<2x8x2x8x2xi32>) -> tensor<256x512x192x1x1xf32>
    %243 = "mhlo.fusion"(%242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x192x1x1xf32>):
      %642 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
      %643 = stablehlo.transpose %642, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
      %644 = mhlo.copy %643 : tensor<512x192x2x2x8x8xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<1x512x49152xf32>
      mhlo.return %645 : tensor<1x512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x192x1x1xf32>) -> tensor<1x512x49152xf32>
    %244 = "mhlo.fusion"(%243, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %245 = "mhlo.fusion"(%244, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %246 = "mhlo.fusion"(%245, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %247 = "mhlo.fusion"(%246, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<256x512x192x1x1xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
      %643 = stablehlo.transpose %642, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
      %644 = mhlo.copy %643 : tensor<512x192x2x2x8x8xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %647 = stablehlo.multiply %arg245, %646 : tensor<512xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.subtract %645, %648 : tensor<512x49152xf32>
      %650 = stablehlo.multiply %649, %649 : tensor<512x49152xf32>
      mhlo.return %650 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<256x512x192x1x1xf32>) -> tensor<512x49152xf32>
    %248 = "mhlo.fusion"(%247, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %249 = "mhlo.fusion"(%248, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %251 = "mhlo.fusion"(%arg85, %250, %246, %242, %arg86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<256x512x192x1x1xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
      %644 = stablehlo.transpose %643, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
      %645 = mhlo.copy %644 : tensor<512x192x2x2x8x8xf32>
      %646 = mhlo.bitcast %645 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %648 = stablehlo.multiply %arg247, %647 : tensor<512xf32>
      %649 = stablehlo.broadcast_in_dim %648, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %650 = stablehlo.subtract %646, %649 : tensor<512x49152xf32>
      %651 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x49152xf32>
      %653 = mhlo.bitcast %652 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %654 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %655 = stablehlo.multiply %653, %654 : tensor<512x192x4x64xf32>
      %656 = stablehlo.add %642, %655 : tensor<512x192x4x64xf32>
      %657 = stablehlo.transpose %656, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %658 = mhlo.copy %657 : tensor<512x4x64x192xf32>
      mhlo.return %658 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x192x1x1xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %252 = mhlo.bitcast %251 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %253 = "mhlo.fusion"(%arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x385xf32>
      mhlo.return %643 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %254 = mhlo.bitcast %253 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %255 = stablehlo.dot %252, %254, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x385xf32>) -> tensor<131072x385xf32>
    %256 = mhlo.bitcast %255 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x385xf32>) -> tensor<512x4x64x385xf32>
    %257 = "mhlo.fusion"(%256, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x385xf32>, %arg246: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      mhlo.return %646 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %258 = "mhlo.fusion"(%257, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
      mhlo.return %642 : tensor<512x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    %259 = "mhlo.fusion"(%258, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %260 = "mhlo.fusion"(%259, %256, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x64xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x64xf32>
      mhlo.return %652 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %261 = "mhlo.fusion"(%260, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %262 = "mhlo.fusion"(%260, %261) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x64xf32>
      mhlo.return %644 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32>
    %263 = "mhlo.fusion"(%262, %256, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x192x4x64xf32>
      mhlo.return %649 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x192x4x64xf32>
    %264 = "mhlo.fusion"(%263, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
      mhlo.return %642 : tensor<512x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    %265 = "mhlo.fusion"(%264, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x192x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
      mhlo.return %642 : tensor<512x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    %266 = "mhlo.fusion"(%265, %256, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x192x4x64xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x192x4x64xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %652 = mhlo.copy %651 : tensor<512x4x64x192xf32>
      mhlo.return %652 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x4x64x192xf32>
    %267 = mhlo.bitcast %266 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %268 = "mhlo.fusion"(%arg89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x192xf32>
      mhlo.return %643 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %269 = mhlo.bitcast %268 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %270 = stablehlo.dot %267, %269, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x192xf32>) -> tensor<131072x192xf32>
    %271 = mhlo.bitcast %270 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %272 = "mhlo.fusion"(%271, %arg90, %242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<256x512x192x1x1xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x512x192x1x1xf32>) -> tensor<2x8x2x8x512x192xf32>
      %647 = stablehlo.transpose %646, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,192,2,2,8,8]{1,0,5,3,4,2}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<512x192x2x2x8x8xf32>
      %648 = mhlo.copy %647 : tensor<512x192x2x2x8x8xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x192x2x2x8x8xf32>) -> tensor<512x192x4x64xf32>
      %650 = stablehlo.add %645, %649 : tensor<512x192x4x64xf32>
      mhlo.return %650 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x192xf32>, tensor<192xf32>, tensor<256x512x192x1x1xf32>) -> tensor<512x192x4x64xf32>
    %273 = mhlo.bitcast %272 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %274 = "mhlo.fusion"(%273, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %275 = "mhlo.fusion"(%274, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %276 = "mhlo.fusion"(%275, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %277 = "mhlo.fusion"(%276, %272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %278 = "mhlo.fusion"(%277, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %279 = "mhlo.fusion"(%278, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %280 = "mhlo.fusion"(%279) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %281 = "mhlo.fusion"(%arg91, %280, %276, %272, %arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %282 = mhlo.bitcast %281 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %283 = "mhlo.fusion"(%arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x384xf32>
      mhlo.return %643 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %284 = mhlo.bitcast %283 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %285 = stablehlo.dot %282, %284, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x384xf32>) -> tensor<131072x384xf32>
    %286 = mhlo.bitcast %285 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x384xf32>) -> tensor<512x4x64x384xf32>
    %287 = "mhlo.fusion"(%286, %arg94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x384xf32>, %arg246: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x384xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
      %647 = stablehlo.negate %645 : tensor<512x384x4x64xf32>
      %648 = stablehlo.exponential %647 : tensor<512x384x4x64xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x384x4x64xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x384x4x64xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x384x4x64xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
      %653 = mhlo.copy %652 : tensor<512x4x64x384xf32>
      mhlo.return %653 : tensor<512x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x384xf32>, tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %288 = mhlo.bitcast %287 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x384xf32>) -> tensor<131072x384xf32>
    %289 = "mhlo.fusion"(%arg95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x384x192xf32>
      mhlo.return %643 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %290 = mhlo.bitcast %289 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %291 = stablehlo.dot %288, %290, precision = [DEFAULT, DEFAULT] : (tensor<131072x384xf32>, tensor<384x192xf32>) -> tensor<131072x192xf32>
    %292 = mhlo.bitcast %291 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %293 = "mhlo.fusion"(%272, %292, %arg96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      mhlo.return %646 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %294 = mhlo.bitcast %293 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %295 = "mhlo.fusion"(%294, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %296 = "mhlo.fusion"(%295, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %297 = "mhlo.fusion"(%296, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %298 = "mhlo.fusion"(%297, %293) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %299 = "mhlo.fusion"(%298, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %300 = "mhlo.fusion"(%299, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %301 = "mhlo.fusion"(%300) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %302 = "mhlo.fusion"(%arg97, %301, %297, %293, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %303 = mhlo.bitcast %302 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %304 = "mhlo.fusion"(%arg99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x385xf32>
      mhlo.return %643 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %305 = mhlo.bitcast %304 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %306 = stablehlo.dot %303, %305, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x385xf32>) -> tensor<131072x385xf32>
    %307 = mhlo.bitcast %306 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x385xf32>) -> tensor<512x4x64x385xf32>
    %308 = "mhlo.fusion"(%307, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x385xf32>, %arg246: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      mhlo.return %646 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %309 = "mhlo.fusion"(%308, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
      mhlo.return %642 : tensor<512x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    %310 = "mhlo.fusion"(%309, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %311 = "mhlo.fusion"(%310, %307, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x64xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x64xf32>
      mhlo.return %652 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %312 = "mhlo.fusion"(%311, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %313 = "mhlo.fusion"(%311, %312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x64xf32>
      mhlo.return %644 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32>
    %314 = "mhlo.fusion"(%313, %307, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x192x4x64xf32>
      mhlo.return %649 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x192x4x64xf32>
    %315 = "mhlo.fusion"(%314, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
      mhlo.return %642 : tensor<512x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    %316 = "mhlo.fusion"(%315, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x192x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
      mhlo.return %642 : tensor<512x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    %317 = "mhlo.fusion"(%316, %307, %arg100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x192x4x64xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x192x4x64xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %652 = mhlo.copy %651 : tensor<512x4x64x192xf32>
      mhlo.return %652 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x4x64x192xf32>
    %318 = mhlo.bitcast %317 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %319 = "mhlo.fusion"(%arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x192xf32>
      mhlo.return %643 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %321 = stablehlo.dot %318, %320, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x192xf32>) -> tensor<131072x192xf32>
    %322 = mhlo.bitcast %321 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %323 = "mhlo.fusion"(%293, %322, %arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      mhlo.return %646 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %324 = mhlo.bitcast %323 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %325 = "mhlo.fusion"(%324, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %326 = "mhlo.fusion"(%325, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %327 = "mhlo.fusion"(%326, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %328 = "mhlo.fusion"(%327, %323) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %329 = "mhlo.fusion"(%328, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %330 = "mhlo.fusion"(%329, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %331 = "mhlo.fusion"(%330) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %332 = "mhlo.fusion"(%arg103, %331, %327, %323, %arg104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %333 = mhlo.bitcast %332 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %334 = "mhlo.fusion"(%arg105) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x384xf32>
      mhlo.return %643 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %335 = mhlo.bitcast %334 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %336 = stablehlo.dot %333, %335, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x384xf32>) -> tensor<131072x384xf32>
    %337 = mhlo.bitcast %336 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x384xf32>) -> tensor<512x4x64x384xf32>
    %338 = "mhlo.fusion"(%337, %arg106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x384xf32>, %arg246: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x384xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
      %647 = stablehlo.negate %645 : tensor<512x384x4x64xf32>
      %648 = stablehlo.exponential %647 : tensor<512x384x4x64xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x384x4x64xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x384x4x64xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x384x4x64xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
      %653 = mhlo.copy %652 : tensor<512x4x64x384xf32>
      mhlo.return %653 : tensor<512x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x384xf32>, tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x384xf32>) -> tensor<131072x384xf32>
    %340 = "mhlo.fusion"(%arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x384x192xf32>
      mhlo.return %643 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %341 = mhlo.bitcast %340 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %342 = stablehlo.dot %339, %341, precision = [DEFAULT, DEFAULT] : (tensor<131072x384xf32>, tensor<384x192xf32>) -> tensor<131072x192xf32>
    %343 = mhlo.bitcast %342 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %344 = "mhlo.fusion"(%323, %343, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      mhlo.return %646 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %345 = mhlo.bitcast %344 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %346 = "mhlo.fusion"(%345, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %347 = "mhlo.fusion"(%346, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %348 = "mhlo.fusion"(%347, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %349 = "mhlo.fusion"(%348, %344) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %350 = "mhlo.fusion"(%349, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %351 = "mhlo.fusion"(%350, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %352 = "mhlo.fusion"(%351) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %353 = "mhlo.fusion"(%arg109, %352, %348, %344, %arg110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %354 = mhlo.bitcast %353 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %355 = "mhlo.fusion"(%arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x385xf32>
      mhlo.return %643 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %356 = mhlo.bitcast %355 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %357 = stablehlo.dot %354, %356, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x385xf32>) -> tensor<131072x385xf32>
    %358 = mhlo.bitcast %357 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x385xf32>) -> tensor<512x4x64x385xf32>
    %359 = "mhlo.fusion"(%358, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x385xf32>, %arg246: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      mhlo.return %646 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %360 = "mhlo.fusion"(%359, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
      mhlo.return %642 : tensor<512x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    %361 = "mhlo.fusion"(%360, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %362 = "mhlo.fusion"(%361, %358, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x64xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x64xf32>
      mhlo.return %652 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %363 = "mhlo.fusion"(%362, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %364 = "mhlo.fusion"(%362, %363) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x64xf32>
      mhlo.return %644 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32>
    %365 = "mhlo.fusion"(%364, %358, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x192x4x64xf32>
      mhlo.return %649 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x192x4x64xf32>
    %366 = "mhlo.fusion"(%365, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
      mhlo.return %642 : tensor<512x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    %367 = "mhlo.fusion"(%366, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x192x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
      mhlo.return %642 : tensor<512x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    %368 = "mhlo.fusion"(%367, %358, %arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x192x4x64xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x192x4x64xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %652 = mhlo.copy %651 : tensor<512x4x64x192xf32>
      mhlo.return %652 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x4x64x192xf32>
    %369 = mhlo.bitcast %368 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %370 = "mhlo.fusion"(%arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x192xf32>
      mhlo.return %643 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %371 = mhlo.bitcast %370 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %372 = stablehlo.dot %369, %371, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x192xf32>) -> tensor<131072x192xf32>
    %373 = mhlo.bitcast %372 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %374 = "mhlo.fusion"(%344, %373, %arg114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      mhlo.return %646 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %375 = mhlo.bitcast %374 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %376 = "mhlo.fusion"(%375, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %377 = "mhlo.fusion"(%376, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %378 = "mhlo.fusion"(%377, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %379 = "mhlo.fusion"(%378, %374) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %380 = "mhlo.fusion"(%379, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %381 = "mhlo.fusion"(%380, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %382 = "mhlo.fusion"(%381) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %383 = "mhlo.fusion"(%arg115, %382, %378, %374, %arg116) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %384 = mhlo.bitcast %383 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %385 = "mhlo.fusion"(%arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x384xf32>
      mhlo.return %643 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %386 = mhlo.bitcast %385 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %387 = stablehlo.dot %384, %386, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x384xf32>) -> tensor<131072x384xf32>
    %388 = mhlo.bitcast %387 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x384xf32>) -> tensor<512x4x64x384xf32>
    %389 = "mhlo.fusion"(%388, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x384xf32>, %arg246: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x384xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
      %647 = stablehlo.negate %645 : tensor<512x384x4x64xf32>
      %648 = stablehlo.exponential %647 : tensor<512x384x4x64xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x384x4x64xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x384x4x64xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x384x4x64xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
      %653 = mhlo.copy %652 : tensor<512x4x64x384xf32>
      mhlo.return %653 : tensor<512x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x384xf32>, tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %390 = mhlo.bitcast %389 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x384xf32>) -> tensor<131072x384xf32>
    %391 = "mhlo.fusion"(%arg119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x384x192xf32>
      mhlo.return %643 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %392 = mhlo.bitcast %391 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %393 = stablehlo.dot %390, %392, precision = [DEFAULT, DEFAULT] : (tensor<131072x384xf32>, tensor<384x192xf32>) -> tensor<131072x192xf32>
    %394 = mhlo.bitcast %393 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %395 = "mhlo.fusion"(%374, %394, %arg120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      mhlo.return %646 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %396 = mhlo.bitcast %395 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %397 = "mhlo.fusion"(%396, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %398 = "mhlo.fusion"(%397, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %399 = "mhlo.fusion"(%398, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %400 = "mhlo.fusion"(%399, %395) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %401 = "mhlo.fusion"(%400, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %402 = "mhlo.fusion"(%401, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %403 = "mhlo.fusion"(%402) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %404 = "mhlo.fusion"(%arg121, %403, %399, %395, %arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %405 = mhlo.bitcast %404 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %406 = "mhlo.fusion"(%arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<385x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,385]{1,0,2,3}"} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x385xf32>
      mhlo.return %643 : tensor<1x1x192x385xf32>
    }) {output_operand_aliasing = []} : (tensor<385x192x1x1xf32>) -> tensor<1x1x192x385xf32>
    %407 = mhlo.bitcast %406 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x385xf32>) -> tensor<192x385xf32>
    %408 = stablehlo.dot %405, %407, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x385xf32>) -> tensor<131072x385xf32>
    %409 = mhlo.bitcast %408 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x385xf32>) -> tensor<512x4x64x385xf32>
    %410 = "mhlo.fusion"(%409, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x385xf32>, %arg246: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      mhlo.return %646 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %411 = "mhlo.fusion"(%410, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.maximum %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
      mhlo.return %642 : tensor<512x1x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4x2xf32>
    %412 = "mhlo.fusion"(%411, %cst_0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x2xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %413 = "mhlo.fusion"(%412, %409, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x1x4x64xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x64xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x64xf32>
      mhlo.return %652 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x1x4x64xf32>
    %414 = "mhlo.fusion"(%413, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %415 = "mhlo.fusion"(%413, %414) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x64xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x64xf32>
      mhlo.return %644 : tensor<512x1x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x64xf32>
    %416 = "mhlo.fusion"(%415, %409, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x64xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 1:193, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x64xf32>) -> tensor<512x4x64xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x64xf32>) -> tensor<512x192x4x64xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x192x4x64xf32>
      mhlo.return %649 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x64xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x192x4x64xf32>
    %417 = "mhlo.fusion"(%416, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
      mhlo.return %642 : tensor<512x192x4x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<f32>) -> tensor<512x192x4x2xf32>
    %418 = "mhlo.fusion"(%417, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x192x4x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
      mhlo.return %642 : tensor<512x192x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x2xf32>, tensor<f32>) -> tensor<512x192x4xf32>
    %419 = "mhlo.fusion"(%418, %409, %arg124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4xf32>, %arg246: tensor<512x4x64x385xf32>, %arg247: tensor<385xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<385xf32>) -> tensor<512x4x64x385xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x385xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,385,4,64]{1,3,2,0}"} : (tensor<512x4x64x385xf32>) -> tensor<512x385x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x385x4x64xf32>
      %646 = stablehlo.slice %645 [0:512, 193:385, 0:4, 0:64] : (tensor<512x385x4x64xf32>) -> tensor<512x192x4x64xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x192x4x64xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x192x4x64xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x192x4xf32>) -> tensor<512x192x4x64xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x192x4x64xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %652 = mhlo.copy %651 : tensor<512x4x64x192xf32>
      mhlo.return %652 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4xf32>, tensor<512x4x64x385xf32>, tensor<385xf32>) -> tensor<512x4x64x192xf32>
    %420 = mhlo.bitcast %419 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %421 = "mhlo.fusion"(%arg125) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,192]{1,0,2,3}"} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x192xf32>
      mhlo.return %643 : tensor<1x1x192x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x192x1x1xf32>) -> tensor<1x1x192x192xf32>
    %422 = mhlo.bitcast %421 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x192xf32>) -> tensor<192x192xf32>
    %423 = stablehlo.dot %420, %422, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x192xf32>) -> tensor<131072x192xf32>
    %424 = mhlo.bitcast %423 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %425 = "mhlo.fusion"(%395, %424, %arg126) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      mhlo.return %646 : tensor<512x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x192x4x64xf32>
    %426 = mhlo.bitcast %425 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
    %427 = "mhlo.fusion"(%426, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %428 = "mhlo.fusion"(%427, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %429 = "mhlo.fusion"(%428, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %430 = "mhlo.fusion"(%429, %425) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x49152xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x49152xf32>
      mhlo.return %647 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
    %431 = "mhlo.fusion"(%430, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %432 = "mhlo.fusion"(%431, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %433 = "mhlo.fusion"(%432) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %434 = "mhlo.fusion"(%arg127, %433, %429, %425, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x49152xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x49152xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x192x4x64xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x192x4x64xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,192]{2,1,3,0}"} : (tensor<512x192x4x64xf32>) -> tensor<512x4x64x192xf32>
      %655 = mhlo.copy %654 : tensor<512x4x64x192xf32>
      mhlo.return %655 : tensor<512x4x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<192xf32>) -> tensor<512x4x64x192xf32>
    %435 = mhlo.bitcast %434 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x192xf32>) -> tensor<131072x192xf32>
    %436 = "mhlo.fusion"(%arg129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x384xf32>
      mhlo.return %643 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %437 = mhlo.bitcast %436 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %438 = stablehlo.dot %435, %437, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x384xf32>) -> tensor<131072x384xf32>
    %439 = mhlo.bitcast %438 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x384xf32>) -> tensor<512x4x64x384xf32>
    %440 = "mhlo.fusion"(%439, %arg130) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x64x384xf32>, %arg246: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x4x64x384xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x64x384xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,384,4,64]{1,3,2,0}"} : (tensor<512x4x64x384xf32>) -> tensor<512x384x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x384x4x64xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x384x4x64xf32>
      %647 = stablehlo.negate %645 : tensor<512x384x4x64xf32>
      %648 = stablehlo.exponential %647 : tensor<512x384x4x64xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x384x4x64xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x384x4x64xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x384x4x64xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,64,384]{2,1,3,0}"} : (tensor<512x384x4x64xf32>) -> tensor<512x4x64x384xf32>
      %653 = mhlo.copy %652 : tensor<512x4x64x384xf32>
      mhlo.return %653 : tensor<512x4x64x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x64x384xf32>, tensor<384xf32>) -> tensor<512x4x64x384xf32>
    %441 = mhlo.bitcast %440 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x64x384xf32>) -> tensor<131072x384xf32>
    %442 = "mhlo.fusion"(%arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192x384x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,192]{1,0,2,3}"} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
      %643 = mhlo.copy %642 : tensor<1x1x384x192xf32>
      mhlo.return %643 : tensor<1x1x384x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x384x1x1xf32>) -> tensor<1x1x384x192xf32>
    %443 = mhlo.bitcast %442 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x192xf32>) -> tensor<384x192xf32>
    %444 = stablehlo.dot %441, %443, precision = [DEFAULT, DEFAULT] : (tensor<131072x384xf32>, tensor<384x192xf32>) -> tensor<131072x192xf32>
    %445 = mhlo.bitcast %444 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x192xf32>) -> tensor<512x4x64x192xf32>
    %446 = "mhlo.fusion"(%425, %445, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x4x64xf32>, %arg246: tensor<512x4x64x192xf32>, %arg247: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x192x4x64xf32>
      %647 = mhlo.bitcast %646 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<1x512x49152xf32>
      mhlo.return %647 : tensor<1x512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<1x512x49152xf32>
    %447 = "mhlo.fusion"(%446, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
      mhlo.return %642 : tensor<1x512x48xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x49152xf32>, tensor<f32>) -> tensor<1x512x48xf32>
    %448 = "mhlo.fusion"(%447, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x48xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [8, 8]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 32>, window_strides = array<i64: 1, 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
      mhlo.return %642 : tensor<1x512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x48xf32>, tensor<f32>) -> tensor<1x512x2xf32>
    %449 = "mhlo.fusion"(%448, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x2xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x2xf32>, tensor<f32>) -> tensor<512xf32>
    %450 = "mhlo.fusion"(%449, %425, %445, %arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x192x4x64xf32>, %arg247: tensor<512x4x64x192xf32>, %arg248: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %643 = stablehlo.add %arg247, %642 : tensor<512x4x64x192xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %645 = mhlo.copy %644 : tensor<512x192x4x64xf32>
      %646 = stablehlo.add %645, %arg246 : tensor<512x192x4x64xf32>
      %647 = mhlo.bitcast %646 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %648 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %649 = stablehlo.multiply %arg245, %648 : tensor<512xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %651 = stablehlo.subtract %647, %650 : tensor<512x49152xf32>
      %652 = stablehlo.multiply %651, %651 : tensor<512x49152xf32>
      mhlo.return %652 : tensor<512x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>) -> tensor<512x49152xf32>
    %451 = "mhlo.fusion"(%450, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x49152xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
      mhlo.return %642 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x49152xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %452 = "mhlo.fusion"(%451, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1536xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [256, 256]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 1024>, window_strides = array<i64: 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
      mhlo.return %642 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>, tensor<f32>) -> tensor<512x2xf32>
    %453 = "mhlo.fusion"(%452) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>):
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %642 = stablehlo.reduce(%arg245 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<512x2xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_2 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %642, %643 : tensor<512xf32>
      %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %645 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %646 = stablehlo.add %644, %645 : tensor<512xf32>
      %647 = stablehlo.rsqrt %646 : tensor<512xf32>
      mhlo.return %647 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>) -> tensor<512xf32>
    %454 = "mhlo.fusion"(%arg133, %453, %449, %425, %445, %arg132, %arg134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x192x4x64xf32>, %arg249: tensor<512x4x64x192xf32>, %arg250: tensor<192xf32>, %arg251: tensor<192xf32>):
      %642 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %643 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<192xf32>) -> tensor<512x4x64x192xf32>
      %644 = stablehlo.add %arg249, %643 : tensor<512x4x64x192xf32>
      %645 = stablehlo.transpose %644, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,192,4,64]{1,3,2,0}"} : (tensor<512x4x64x192xf32>) -> tensor<512x192x4x64xf32>
      %646 = mhlo.copy %645 : tensor<512x192x4x64xf32>
      %647 = stablehlo.add %646, %arg248 : tensor<512x192x4x64xf32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x49152xf32>
      %cst_1 = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %649 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %650 = stablehlo.multiply %arg247, %649 : tensor<512xf32>
      %651 = stablehlo.broadcast_in_dim %650, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %652 = stablehlo.subtract %648, %651 : tensor<512x49152xf32>
      %653 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x49152xf32>
      %654 = stablehlo.multiply %652, %653 : tensor<512x49152xf32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x49152xf32>) -> tensor<512x192x4x64xf32>
      %656 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<512x192x4x64xf32>
      %657 = stablehlo.multiply %655, %656 : tensor<512x192x4x64xf32>
      %658 = stablehlo.add %642, %657 : tensor<512x192x4x64xf32>
      %659 = mhlo.bitcast %658 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x192x4x64xf32>) -> tensor<512x192x2x2x8x8xf32>
      %660 = stablehlo.transpose %659, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,8,2,8,512,192]{3,1,2,0,5,4}"} : (tensor<512x192x2x2x8x8xf32>) -> tensor<2x8x2x8x512x192xf32>
      %661 = mhlo.copy %660 : tensor<2x8x2x8x512x192xf32>
      %662 = mhlo.bitcast %661 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[256,512,192,1,1]{2,1,0,4,3}"} : (tensor<2x8x2x8x512x192xf32>) -> tensor<256x512x192x1x1xf32>
      %663 = mhlo.copy %662 : tensor<256x512x192x1x1xf32>
      mhlo.return %663 : tensor<256x512x192x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x192x4x64xf32>, tensor<512x4x64x192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<256x512x192x1x1xf32>
    %455 = "mhlo.fusion"(%8, %9, %454) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x16x16xf32>, %arg246: tensor<256x2xi32>, %arg247: tensor<256x512x192x1x1xf32>):
      %642 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %643 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x192x16x16xf32>, tensor<256x2xi32>, tensor<256x512x192x1x1xf32>) -> tensor<512x192x16x16xf32>
      mhlo.return %642 : tensor<512x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x16x16xf32>, tensor<256x2xi32>, tensor<256x512x192x1x1xf32>) -> tensor<512x192x16x16xf32>
    %456 = "mhlo.fusion"(%455) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x192x16x16xf32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,192]{2,1,3,0}"} : (tensor<512x192x16x16xf32>) -> tensor<512x16x16x192xf32>
      %643 = mhlo.copy %642 : tensor<512x16x16x192xf32>
      mhlo.return %643 : tensor<512x16x16x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x192x16x16xf32>) -> tensor<512x16x16x192xf32>
    %457 = mhlo.bitcast %456 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x16x16x192xf32>) -> tensor<131072x192xf32>
    %458 = "mhlo.fusion"(%arg135) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384x192x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,192,384]{1,0,2,3}"} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
      %643 = mhlo.copy %642 : tensor<1x1x192x384xf32>
      mhlo.return %643 : tensor<1x1x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x1x1xf32>) -> tensor<1x1x192x384xf32>
    %459 = mhlo.bitcast %458 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x192x384xf32>) -> tensor<192x384xf32>
    %460 = stablehlo.dot %457, %459, precision = [DEFAULT, DEFAULT] : (tensor<131072x192xf32>, tensor<192x384xf32>) -> tensor<131072x384xf32>
    %461 = mhlo.bitcast %460 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x384xf32>) -> tensor<512x16x16x384xf32>
    %462 = "mhlo.fusion"(%arg137, %arg136, %7, %461, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<512x16x16x384xf32>, %arg249: tensor<384xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x16x16x384xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x16x16x384xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x16x16x384xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<384xf32>) -> tensor<512x16x16x384xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x16x16x384xf32>
      mhlo.return %649 : tensor<512x16x16x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<512x16x16x384xf32>, tensor<384xf32>) -> tensor<512x16x16x384xf32>
    %463 = mhlo.bitcast %462 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x16x16x384xf32>) -> tensor<131072x384xf32>
    %464 = "mhlo.fusion"(%arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768x384x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,384,768]{1,0,2,3}"} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
      %643 = mhlo.copy %642 : tensor<1x1x384x768xf32>
      mhlo.return %643 : tensor<1x1x384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x384x1x1xf32>) -> tensor<1x1x384x768xf32>
    %465 = mhlo.bitcast %464 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x384x768xf32>) -> tensor<384x768xf32>
    %466 = stablehlo.dot %463, %465, precision = [DEFAULT, DEFAULT] : (tensor<131072x384xf32>, tensor<384x768xf32>) -> tensor<131072x768xf32>
    %467 = mhlo.bitcast %466 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<131072x768xf32>) -> tensor<512x16x16x768xf32>
    %468 = "mhlo.fusion"(%arg140, %arg139, %6, %467, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<512x16x16x768xf32>, %arg249: tensor<768xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x16x16x768xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x16x16x768xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x16x16x768xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<768xf32>) -> tensor<512x16x16x768xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x16x16x768xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,768,16,16]{1,3,2,0}"} : (tensor<512x16x16x768xf32>) -> tensor<512x768x16x16xf32>
      %651 = mhlo.copy %650 : tensor<512x768x16x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x768x16x16xf32>
      %653 = stablehlo.negate %651 : tensor<512x768x16x16xf32>
      %654 = stablehlo.exponential %653 : tensor<512x768x16x16xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x768x16x16xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x768x16x16xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x768x16x16xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,16,16,768]{2,1,3,0}"} : (tensor<512x768x16x16xf32>) -> tensor<512x16x16x768xf32>
      %659 = mhlo.copy %658 : tensor<512x16x16x768xf32>
      mhlo.return %659 : tensor<512x16x16x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<512x16x16x768xf32>, tensor<768xf32>) -> tensor<512x16x16x768xf32>
    %469 = "mhlo.fusion"(%arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,768]{1,0,2,3}"} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x768xf32>
      mhlo.return %643 : tensor<3x3x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x3x3xf32>) -> tensor<3x3x1x768xf32>
    %470 = stablehlo.convolution(%468, %469) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x16x16x768xf32>, tensor<3x3x1x768xf32>) -> tensor<512x8x8x768xf32>
    %471 = "mhlo.fusion"(%arg143, %arg142, %5, %470, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<512x8x8x768xf32>, %arg249: tensor<768xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x8x8x768xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x8x8x768xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x8x8x768xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<768xf32>) -> tensor<512x8x8x768xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x8x8x768xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,768,8,8]{1,3,2,0}"} : (tensor<512x8x8x768xf32>) -> tensor<512x768x8x8xf32>
      %651 = mhlo.copy %650 : tensor<512x768x8x8xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x768x8x8xf32>
      %653 = stablehlo.negate %651 : tensor<512x768x8x8xf32>
      %654 = stablehlo.exponential %653 : tensor<512x768x8x8xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x768x8x8xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x768x8x8xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x768x8x8xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,8,8,768]{2,1,3,0}"} : (tensor<512x768x8x8xf32>) -> tensor<512x8x8x768xf32>
      %659 = mhlo.copy %658 : tensor<512x8x8x768xf32>
      mhlo.return %659 : tensor<512x8x8x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<512x8x8x768xf32>, tensor<768xf32>) -> tensor<512x8x8x768xf32>
    %472 = mhlo.bitcast %471 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x8x8x768xf32>) -> tensor<32768x768xf32>
    %473 = "mhlo.fusion"(%arg144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x768x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,768,512]{1,0,2,3}"} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
      %643 = mhlo.copy %642 : tensor<1x1x768x512xf32>
      mhlo.return %643 : tensor<1x1x768x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768x1x1xf32>) -> tensor<1x1x768x512xf32>
    %474 = mhlo.bitcast %473 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x768x512xf32>) -> tensor<768x512xf32>
    %475 = stablehlo.dot %472, %474, precision = [DEFAULT, DEFAULT] : (tensor<32768x768xf32>, tensor<768x512xf32>) -> tensor<32768x512xf32>
    %476 = mhlo.bitcast %475 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x512xf32>) -> tensor<512x8x8x512xf32>
    %477 = "mhlo.fusion"(%arg146, %arg145, %4, %476, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x8x8x512xf32>, %arg249: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x8x8x512xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x8x8x512xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x8x8x512xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x8x8x512xf32>
      mhlo.return %649 : tensor<512x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x8x8x512xf32>, tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %478 = "mhlo.fusion"(%arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x3x3xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[3,3,1,512]{1,0,2,3}"} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
      %643 = mhlo.copy %642 : tensor<3x3x1x512xf32>
      mhlo.return %643 : tensor<3x3x1x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x3x3xf32>) -> tensor<3x3x1x512xf32>
    %479 = stablehlo.convolution(%477, %478) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 512 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x8x8x512xf32>, tensor<3x3x1x512xf32>) -> tensor<512x8x8x512xf32>
    %480 = "mhlo.fusion"(%arg149, %arg148, %3, %479, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x8x8x512xf32>, %arg249: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x8x8x512xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x8x8x512xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x8x8x512xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x8x8x512xf32>
      %650 = stablehlo.transpose %649, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,8,8]{1,3,2,0}"} : (tensor<512x8x8x512xf32>) -> tensor<512x512x8x8xf32>
      %651 = mhlo.copy %650 : tensor<512x512x8x8xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %652 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512x8x8xf32>
      %653 = stablehlo.negate %651 : tensor<512x512x8x8xf32>
      %654 = stablehlo.exponential %653 : tensor<512x512x8x8xf32>
      %655 = stablehlo.add %652, %654 : tensor<512x512x8x8xf32>
      %656 = stablehlo.divide %652, %655 : tensor<512x512x8x8xf32>
      %657 = stablehlo.multiply %651, %656 : tensor<512x512x8x8xf32>
      %658 = stablehlo.transpose %657, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,8,8,512]{2,1,3,0}"} : (tensor<512x512x8x8xf32>) -> tensor<512x8x8x512xf32>
      %659 = mhlo.copy %658 : tensor<512x8x8x512xf32>
      mhlo.return %659 : tensor<512x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x8x8x512xf32>, tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %481 = mhlo.bitcast %480 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x8x8x512xf32>) -> tensor<32768x512xf32>
    %482 = "mhlo.fusion"(%arg150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x512x256xf32>
      mhlo.return %643 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %483 = mhlo.bitcast %482 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %484 = stablehlo.dot %481, %483, precision = [DEFAULT, DEFAULT] : (tensor<32768x512xf32>, tensor<512x256xf32>) -> tensor<32768x256xf32>
    %485 = mhlo.bitcast %484 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x8x8x256xf32>
    %486 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %642 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %643 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %644 = stablehlo.multiply %642, %643 : tensor<4xi32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %646 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %647 = stablehlo.add %645, %646 : tensor<2x4xi32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0, 3, 2]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "s32[2,4,1,1]{1,0,3,2}"} : (tensor<2x4xi32>) -> tensor<2x4x1x1xi32>
      %649 = mhlo.copy %648 : tensor<2x4x1x1xi32>
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      %650 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %651 = stablehlo.compare LT, %649, %650 : (tensor<2x4x1x1xi32>, tensor<2x4x1x1xi32>) -> tensor<2x4x1x1xi1>
      %c_2 = stablehlo.constant dense<8> : tensor<i32>
      %652 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4x1x1xi32>
      %653 = stablehlo.add %649, %652 : tensor<2x4x1x1xi32>
      %654 = stablehlo.select %651, %653, %649 : tensor<2x4x1x1xi1>, tensor<2x4x1x1xi32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<2x4x1x1xi32>) -> tensor<2x4xi32>
      %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %657 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %658 = stablehlo.compare LT, %647, %657 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %659 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %660 = stablehlo.add %647, %659 : tensor<2x4xi32>
      %661 = stablehlo.select %658, %660, %647 : tensor<2x4xi1>, tensor<2x4xi32>
      %662 = stablehlo.broadcast_in_dim %661, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %663 = stablehlo.concatenate %656, %662, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      mhlo.return %663 : tensor<2x4x2x4x2xi32>
    }) {output_operand_aliasing = []} : () -> tensor<2x4x2x4x2xi32>
    %487 = "mhlo.fusion"(%485, %486) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x8x8x256xf32>, %arg246: tensor<2x4x2x4x2xi32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,8,8]{1,3,2,0}"} : (tensor<512x8x8x256xf32>) -> tensor<512x256x8x8xf32>
      %643 = mhlo.copy %642 : tensor<512x256x8x8xf32>
      %644 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %645 = "stablehlo.gather"(%643, %644) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 512, 256, 1, 1>}> : (tensor<512x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x512x256x1x1xf32>
      mhlo.return %645 : tensor<64x512x256x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x8x8x256xf32>, tensor<2x4x2x4x2xi32>) -> tensor<64x512x256x1x1xf32>
    %488 = "mhlo.fusion"(%487) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x512x256x1x1xf32>):
      %642 = mhlo.bitcast %arg245 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
      %643 = stablehlo.transpose %642, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
      %644 = mhlo.copy %643 : tensor<512x256x2x2x4x4xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<1x512x16384xf32>
      mhlo.return %645 : tensor<1x512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x256x1x1xf32>) -> tensor<1x512x16384xf32>
    %489 = "mhlo.fusion"(%488, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %490 = "mhlo.fusion"(%489, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %491 = "mhlo.fusion"(%490, %487) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<64x512x256x1x1xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
      %643 = stablehlo.transpose %642, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
      %644 = mhlo.copy %643 : tensor<512x256x2x2x4x4xf32>
      %645 = mhlo.bitcast %644 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %647 = stablehlo.multiply %arg245, %646 : tensor<512xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %649 = stablehlo.subtract %645, %648 : tensor<512x16384xf32>
      %650 = stablehlo.multiply %649, %649 : tensor<512x16384xf32>
      mhlo.return %650 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<64x512x256x1x1xf32>) -> tensor<512x16384xf32>
    %492 = "mhlo.fusion"(%491, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %493 = "mhlo.fusion"(%492, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %494 = "mhlo.fusion"(%493) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %495 = "mhlo.fusion"(%arg151, %494, %490, %487, %arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<64x512x256x1x1xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
      %644 = stablehlo.transpose %643, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
      %645 = mhlo.copy %644 : tensor<512x256x2x2x4x4xf32>
      %646 = mhlo.bitcast %645 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %648 = stablehlo.multiply %arg247, %647 : tensor<512xf32>
      %649 = stablehlo.broadcast_in_dim %648, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %650 = stablehlo.subtract %646, %649 : tensor<512x16384xf32>
      %651 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x16384xf32>
      %653 = mhlo.bitcast %652 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %654 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %655 = stablehlo.multiply %653, %654 : tensor<512x256x4x16xf32>
      %656 = stablehlo.add %642, %655 : tensor<512x256x4x16xf32>
      %657 = stablehlo.transpose %656, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %658 = mhlo.copy %657 : tensor<512x4x16x256xf32>
      mhlo.return %658 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x256x1x1xf32>, tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %496 = mhlo.bitcast %495 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %497 = "mhlo.fusion"(%arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x513xf32>
      mhlo.return %643 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %498 = mhlo.bitcast %497 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %499 = stablehlo.dot %496, %498, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x513xf32>) -> tensor<32768x513xf32>
    %500 = mhlo.bitcast %499 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x513xf32>) -> tensor<512x4x16x513xf32>
    %501 = "mhlo.fusion"(%500, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x513xf32>, %arg246: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.reduce(%646 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %647 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x1x4xf32>
    %502 = "mhlo.fusion"(%501, %500, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x16xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x16xf32>
      mhlo.return %652 : tensor<512x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x1x4x16xf32>
    %503 = "mhlo.fusion"(%502, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %504 = "mhlo.fusion"(%502, %503) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x16xf32>
      mhlo.return %644 : tensor<512x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x16xf32>
    %505 = "mhlo.fusion"(%504, %500, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 1:257, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x16xf32>) -> tensor<512x4x16xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x16xf32>) -> tensor<512x256x4x16xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %650 = stablehlo.reduce(%649 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<512x256x4x16xf32>, tensor<f32>) -> tensor<512x256x4xf32>
      mhlo.return %650 : tensor<512x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x256x4xf32>
    %506 = "mhlo.fusion"(%505, %500, %arg154) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 257:513, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x4x16xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x256x4x16xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x256x4xf32>) -> tensor<512x256x4x16xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x256x4x16xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %652 = mhlo.copy %651 : tensor<512x4x16x256xf32>
      mhlo.return %652 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x4x16x256xf32>
    %507 = mhlo.bitcast %506 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %508 = "mhlo.fusion"(%arg155) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x256xf32>
      mhlo.return %643 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %509 = mhlo.bitcast %508 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %510 = stablehlo.dot %507, %509, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %511 = mhlo.bitcast %510 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x4x16x256xf32>
    %512 = "mhlo.fusion"(%511, %arg156, %487) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x512x256x1x1xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = mhlo.bitcast %arg247 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x512x256x1x1xf32>) -> tensor<2x4x2x4x512x256xf32>
      %647 = stablehlo.transpose %646, dims = [4, 5, 0, 2, 1, 3] {result_layout = dense<[1, 0, 5, 3, 4, 2]> : tensor<6xindex>, xla_shape = "f32[512,256,2,2,4,4]{1,0,5,3,4,2}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<512x256x2x2x4x4xf32>
      %648 = mhlo.copy %647 : tensor<512x256x2x2x4x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<512x256x2x2x4x4xf32>) -> tensor<512x256x4x16xf32>
      %650 = stablehlo.add %645, %649 : tensor<512x256x4x16xf32>
      mhlo.return %650 : tensor<512x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x256xf32>, tensor<256xf32>, tensor<64x512x256x1x1xf32>) -> tensor<512x256x4x16xf32>
    %513 = mhlo.bitcast %512 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
    %514 = "mhlo.fusion"(%513, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %515 = "mhlo.fusion"(%514, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %516 = "mhlo.fusion"(%515, %512) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x256x4x16xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x16384xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x16384xf32>
      mhlo.return %647 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %517 = "mhlo.fusion"(%516, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %518 = "mhlo.fusion"(%517, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %519 = "mhlo.fusion"(%518) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %520 = "mhlo.fusion"(%arg157, %519, %515, %512, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x256x4x16xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x16384xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x16384xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x256x4x16xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x256x4x16xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %655 = mhlo.copy %654 : tensor<512x4x16x256xf32>
      mhlo.return %655 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %521 = mhlo.bitcast %520 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %522 = "mhlo.fusion"(%arg159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x512xf32>
      mhlo.return %643 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %523 = mhlo.bitcast %522 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %524 = stablehlo.dot %521, %523, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x512xf32>) -> tensor<32768x512xf32>
    %525 = mhlo.bitcast %524 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x512xf32>) -> tensor<512x4x16x512xf32>
    %526 = "mhlo.fusion"(%525, %arg160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x512xf32>, %arg246: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x4x16x512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x512xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,4,16]{1,3,2,0}"} : (tensor<512x4x16x512xf32>) -> tensor<512x512x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512x4x16xf32>
      %647 = stablehlo.negate %645 : tensor<512x512x4x16xf32>
      %648 = stablehlo.exponential %647 : tensor<512x512x4x16xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x512x4x16xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x512x4x16xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x512x4x16xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,512]{2,1,3,0}"} : (tensor<512x512x4x16xf32>) -> tensor<512x4x16x512xf32>
      %653 = mhlo.copy %652 : tensor<512x4x16x512xf32>
      mhlo.return %653 : tensor<512x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x512xf32>, tensor<512xf32>) -> tensor<512x4x16x512xf32>
    %527 = mhlo.bitcast %526 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x512xf32>) -> tensor<32768x512xf32>
    %528 = "mhlo.fusion"(%arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x512x256xf32>
      mhlo.return %643 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %529 = mhlo.bitcast %528 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %530 = stablehlo.dot %527, %529, precision = [DEFAULT, DEFAULT] : (tensor<32768x512xf32>, tensor<512x256xf32>) -> tensor<32768x256xf32>
    %531 = mhlo.bitcast %530 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x4x16x256xf32>
    %532 = "mhlo.fusion"(%512, %531, %arg162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4x16xf32>, %arg246: tensor<512x4x16x256xf32>, %arg247: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x256x4x16xf32>
      mhlo.return %646 : tensor<512x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %533 = mhlo.bitcast %532 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
    %534 = "mhlo.fusion"(%533, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %535 = "mhlo.fusion"(%534, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %536 = "mhlo.fusion"(%535, %532) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x256x4x16xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x16384xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x16384xf32>
      mhlo.return %647 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %537 = "mhlo.fusion"(%536, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %538 = "mhlo.fusion"(%537, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %539 = "mhlo.fusion"(%538) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %540 = "mhlo.fusion"(%arg163, %539, %535, %532, %arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x256x4x16xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x16384xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x16384xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x256x4x16xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x256x4x16xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %655 = mhlo.copy %654 : tensor<512x4x16x256xf32>
      mhlo.return %655 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %541 = mhlo.bitcast %540 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %542 = "mhlo.fusion"(%arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x513xf32>
      mhlo.return %643 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %543 = mhlo.bitcast %542 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %544 = stablehlo.dot %541, %543, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x513xf32>) -> tensor<32768x513xf32>
    %545 = mhlo.bitcast %544 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x513xf32>) -> tensor<512x4x16x513xf32>
    %546 = "mhlo.fusion"(%545, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x513xf32>, %arg246: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.reduce(%646 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %647 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x1x4xf32>
    %547 = "mhlo.fusion"(%546, %545, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x16xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x16xf32>
      mhlo.return %652 : tensor<512x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x1x4x16xf32>
    %548 = "mhlo.fusion"(%547, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %549 = "mhlo.fusion"(%547, %548) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x16xf32>
      mhlo.return %644 : tensor<512x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x16xf32>
    %550 = "mhlo.fusion"(%549, %545, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 1:257, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x16xf32>) -> tensor<512x4x16xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x16xf32>) -> tensor<512x256x4x16xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %650 = stablehlo.reduce(%649 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<512x256x4x16xf32>, tensor<f32>) -> tensor<512x256x4xf32>
      mhlo.return %650 : tensor<512x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x256x4xf32>
    %551 = "mhlo.fusion"(%550, %545, %arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 257:513, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x4x16xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x256x4x16xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x256x4xf32>) -> tensor<512x256x4x16xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x256x4x16xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %652 = mhlo.copy %651 : tensor<512x4x16x256xf32>
      mhlo.return %652 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x4x16x256xf32>
    %552 = mhlo.bitcast %551 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %553 = "mhlo.fusion"(%arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x256xf32>
      mhlo.return %643 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %554 = mhlo.bitcast %553 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %555 = stablehlo.dot %552, %554, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %556 = mhlo.bitcast %555 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x4x16x256xf32>
    %557 = "mhlo.fusion"(%532, %556, %arg168) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4x16xf32>, %arg246: tensor<512x4x16x256xf32>, %arg247: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x256x4x16xf32>
      mhlo.return %646 : tensor<512x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %558 = mhlo.bitcast %557 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
    %559 = "mhlo.fusion"(%558, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %560 = "mhlo.fusion"(%559, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %561 = "mhlo.fusion"(%560, %557) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x256x4x16xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x16384xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x16384xf32>
      mhlo.return %647 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %562 = "mhlo.fusion"(%561, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %563 = "mhlo.fusion"(%562, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %564 = "mhlo.fusion"(%563) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %565 = "mhlo.fusion"(%arg169, %564, %560, %557, %arg170) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x256x4x16xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x16384xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x16384xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x256x4x16xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x256x4x16xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %655 = mhlo.copy %654 : tensor<512x4x16x256xf32>
      mhlo.return %655 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %566 = mhlo.bitcast %565 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %567 = "mhlo.fusion"(%arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x512xf32>
      mhlo.return %643 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %568 = mhlo.bitcast %567 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %569 = stablehlo.dot %566, %568, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x512xf32>) -> tensor<32768x512xf32>
    %570 = mhlo.bitcast %569 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x512xf32>) -> tensor<512x4x16x512xf32>
    %571 = "mhlo.fusion"(%570, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x512xf32>, %arg246: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x4x16x512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x512xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,4,16]{1,3,2,0}"} : (tensor<512x4x16x512xf32>) -> tensor<512x512x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512x4x16xf32>
      %647 = stablehlo.negate %645 : tensor<512x512x4x16xf32>
      %648 = stablehlo.exponential %647 : tensor<512x512x4x16xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x512x4x16xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x512x4x16xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x512x4x16xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,512]{2,1,3,0}"} : (tensor<512x512x4x16xf32>) -> tensor<512x4x16x512xf32>
      %653 = mhlo.copy %652 : tensor<512x4x16x512xf32>
      mhlo.return %653 : tensor<512x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x512xf32>, tensor<512xf32>) -> tensor<512x4x16x512xf32>
    %572 = mhlo.bitcast %571 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x512xf32>) -> tensor<32768x512xf32>
    %573 = "mhlo.fusion"(%arg173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x512x256xf32>
      mhlo.return %643 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %574 = mhlo.bitcast %573 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %575 = stablehlo.dot %572, %574, precision = [DEFAULT, DEFAULT] : (tensor<32768x512xf32>, tensor<512x256xf32>) -> tensor<32768x256xf32>
    %576 = mhlo.bitcast %575 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x4x16x256xf32>
    %577 = "mhlo.fusion"(%557, %576, %arg174) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4x16xf32>, %arg246: tensor<512x4x16x256xf32>, %arg247: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x256x4x16xf32>
      mhlo.return %646 : tensor<512x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %578 = mhlo.bitcast %577 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
    %579 = "mhlo.fusion"(%578, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %580 = "mhlo.fusion"(%579, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %581 = "mhlo.fusion"(%580, %577) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x256x4x16xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x16384xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x16384xf32>
      mhlo.return %647 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %582 = "mhlo.fusion"(%581, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %583 = "mhlo.fusion"(%582, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %584 = "mhlo.fusion"(%583) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %585 = "mhlo.fusion"(%arg175, %584, %580, %577, %arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x256x4x16xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x16384xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x16384xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x256x4x16xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x256x4x16xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %655 = mhlo.copy %654 : tensor<512x4x16x256xf32>
      mhlo.return %655 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %586 = mhlo.bitcast %585 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %587 = "mhlo.fusion"(%arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<513x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,513]{1,0,2,3}"} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x513xf32>
      mhlo.return %643 : tensor<1x1x256x513xf32>
    }) {output_operand_aliasing = []} : (tensor<513x256x1x1xf32>) -> tensor<1x1x256x513xf32>
    %588 = mhlo.bitcast %587 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x513xf32>) -> tensor<256x513xf32>
    %589 = stablehlo.dot %586, %588, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x513xf32>) -> tensor<32768x513xf32>
    %590 = mhlo.bitcast %589 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x513xf32>) -> tensor<512x4x16x513xf32>
    %591 = "mhlo.fusion"(%590, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x513xf32>, %arg246: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.reduce(%646 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %647 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x1x4xf32>
    %592 = "mhlo.fusion"(%591, %590, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 0:1, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x1x4x16xf32>
      %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x4xf32>
      %648 = stablehlo.maximum %647, %arg245 : tensor<512x1x4xf32>
      %649 = mhlo.bitcast %648 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
      %651 = stablehlo.subtract %646, %650 : tensor<512x1x4x16xf32>
      %652 = stablehlo.exponential %651 : tensor<512x1x4x16xf32>
      mhlo.return %652 : tensor<512x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x1x4x16xf32>
    %593 = "mhlo.fusion"(%592, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [3] : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
      mhlo.return %642 : tensor<512x1x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<f32>) -> tensor<512x1x4xf32>
    %594 = "mhlo.fusion"(%592, %593) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<512x1x4xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x4xf32>) -> tensor<512x4xf32>
      %643 = stablehlo.broadcast_in_dim %642, dims = [0, 2] : (tensor<512x4xf32>) -> tensor<512x1x4x16xf32>
      %644 = stablehlo.divide %arg245, %643 : tensor<512x1x4x16xf32>
      mhlo.return %644 : tensor<512x1x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<512x1x4xf32>) -> tensor<512x1x4x16xf32>
    %595 = "mhlo.fusion"(%594, %590, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x1x4x16xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 1:257, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
      %647 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x4x16xf32>) -> tensor<512x4x16xf32>
      %648 = stablehlo.broadcast_in_dim %647, dims = [0, 2, 3] : (tensor<512x4x16xf32>) -> tensor<512x256x4x16xf32>
      %649 = stablehlo.multiply %646, %648 : tensor<512x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %650 = stablehlo.reduce(%649 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<512x256x4x16xf32>, tensor<f32>) -> tensor<512x256x4xf32>
      mhlo.return %650 : tensor<512x256x4xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x4x16xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x256x4xf32>
    %596 = "mhlo.fusion"(%595, %590, %arg178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4xf32>, %arg246: tensor<512x4x16x513xf32>, %arg247: tensor<513xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<513xf32>) -> tensor<512x4x16x513xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x513xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,513,4,16]{1,3,2,0}"} : (tensor<512x4x16x513xf32>) -> tensor<512x513x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x513x4x16xf32>
      %646 = stablehlo.slice %645 [0:512, 257:513, 0:4, 0:16] : (tensor<512x513x4x16xf32>) -> tensor<512x256x4x16xf32>
      %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %647 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x256x4x16xf32>
      %648 = stablehlo.maximum %646, %647 : tensor<512x256x4x16xf32>
      %649 = stablehlo.broadcast_in_dim %arg245, dims = [0, 1, 2] : (tensor<512x256x4xf32>) -> tensor<512x256x4x16xf32>
      %650 = stablehlo.multiply %648, %649 : tensor<512x256x4x16xf32>
      %651 = stablehlo.transpose %650, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %652 = mhlo.copy %651 : tensor<512x4x16x256xf32>
      mhlo.return %652 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4xf32>, tensor<512x4x16x513xf32>, tensor<513xf32>) -> tensor<512x4x16x256xf32>
    %597 = mhlo.bitcast %596 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %598 = "mhlo.fusion"(%arg179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,256]{1,0,2,3}"} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x256xf32>
      mhlo.return %643 : tensor<1x1x256x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x256x1x1xf32>) -> tensor<1x1x256x256xf32>
    %599 = mhlo.bitcast %598 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x256xf32>) -> tensor<256x256xf32>
    %600 = stablehlo.dot %597, %599, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x256xf32>) -> tensor<32768x256xf32>
    %601 = mhlo.bitcast %600 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x4x16x256xf32>
    %602 = "mhlo.fusion"(%577, %601, %arg180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4x16xf32>, %arg246: tensor<512x4x16x256xf32>, %arg247: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x256x4x16xf32>
      mhlo.return %646 : tensor<512x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>) -> tensor<512x256x4x16xf32>
    %603 = mhlo.bitcast %602 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
    %604 = "mhlo.fusion"(%603, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %605 = "mhlo.fusion"(%604, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %606 = "mhlo.fusion"(%605, %602) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x256x4x16xf32>):
      %642 = mhlo.bitcast %arg246 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %643 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %644 = stablehlo.multiply %arg245, %643 : tensor<512xf32>
      %645 = stablehlo.broadcast_in_dim %644, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %646 = stablehlo.subtract %642, %645 : tensor<512x16384xf32>
      %647 = stablehlo.multiply %646, %646 : tensor<512x16384xf32>
      mhlo.return %647 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
    %607 = "mhlo.fusion"(%606, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %608 = "mhlo.fusion"(%607, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %609 = "mhlo.fusion"(%608) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %610 = "mhlo.fusion"(%arg181, %609, %605, %602, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x256x4x16xf32>, %arg249: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = mhlo.bitcast %arg248 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.multiply %arg247, %644 : tensor<512xf32>
      %646 = stablehlo.broadcast_in_dim %645, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %647 = stablehlo.subtract %643, %646 : tensor<512x16384xf32>
      %648 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %649 = stablehlo.multiply %647, %648 : tensor<512x16384xf32>
      %650 = mhlo.bitcast %649 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %651 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %652 = stablehlo.multiply %650, %651 : tensor<512x256x4x16xf32>
      %653 = stablehlo.add %642, %652 : tensor<512x256x4x16xf32>
      %654 = stablehlo.transpose %653, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,256]{2,1,3,0}"} : (tensor<512x256x4x16xf32>) -> tensor<512x4x16x256xf32>
      %655 = mhlo.copy %654 : tensor<512x4x16x256xf32>
      mhlo.return %655 : tensor<512x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<256xf32>) -> tensor<512x4x16x256xf32>
    %611 = mhlo.bitcast %610 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x256xf32>) -> tensor<32768x256xf32>
    %612 = "mhlo.fusion"(%arg183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x512xf32>
      mhlo.return %643 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %613 = mhlo.bitcast %612 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %614 = stablehlo.dot %611, %613, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x512xf32>) -> tensor<32768x512xf32>
    %615 = mhlo.bitcast %614 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x512xf32>) -> tensor<512x4x16x512xf32>
    %616 = "mhlo.fusion"(%615, %arg184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x4x16x512xf32>, %arg246: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x4x16x512xf32>
      %643 = stablehlo.add %arg245, %642 : tensor<512x4x16x512xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,512,4,16]{1,3,2,0}"} : (tensor<512x4x16x512xf32>) -> tensor<512x512x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x512x4x16xf32>
      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %646 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512x4x16xf32>
      %647 = stablehlo.negate %645 : tensor<512x512x4x16xf32>
      %648 = stablehlo.exponential %647 : tensor<512x512x4x16xf32>
      %649 = stablehlo.add %646, %648 : tensor<512x512x4x16xf32>
      %650 = stablehlo.divide %646, %649 : tensor<512x512x4x16xf32>
      %651 = stablehlo.multiply %645, %650 : tensor<512x512x4x16xf32>
      %652 = stablehlo.transpose %651, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,4,16,512]{2,1,3,0}"} : (tensor<512x512x4x16xf32>) -> tensor<512x4x16x512xf32>
      %653 = mhlo.copy %652 : tensor<512x4x16x512xf32>
      mhlo.return %653 : tensor<512x4x16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x4x16x512xf32>, tensor<512xf32>) -> tensor<512x4x16x512xf32>
    %617 = mhlo.bitcast %616 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x4x16x512xf32>) -> tensor<32768x512xf32>
    %618 = "mhlo.fusion"(%arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256x512x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,512,256]{1,0,2,3}"} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
      %643 = mhlo.copy %642 : tensor<1x1x512x256xf32>
      mhlo.return %643 : tensor<1x1x512x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256x512x1x1xf32>) -> tensor<1x1x512x256xf32>
    %619 = mhlo.bitcast %618 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x512x256xf32>) -> tensor<512x256xf32>
    %620 = stablehlo.dot %617, %619, precision = [DEFAULT, DEFAULT] : (tensor<32768x512xf32>, tensor<512x256xf32>) -> tensor<32768x256xf32>
    %621 = mhlo.bitcast %620 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x256xf32>) -> tensor<512x4x16x256xf32>
    %622 = "mhlo.fusion"(%602, %621, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x4x16xf32>, %arg246: tensor<512x4x16x256xf32>, %arg247: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg246, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = stablehlo.add %645, %arg245 : tensor<512x256x4x16xf32>
      %647 = mhlo.bitcast %646 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<1x512x16384xf32>
      mhlo.return %647 : tensor<1x512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>) -> tensor<1x512x16384xf32>
    %623 = "mhlo.fusion"(%622, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<0> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1024>, window_strides = array<i64: 1, 1, 1024>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
      mhlo.return %642 : tensor<1x512x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16384xf32>, tensor<f32>) -> tensor<1x512x16xf32>
    %624 = "mhlo.fusion"(%623, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1x512x16xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [0, 2] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %625 = "mhlo.fusion"(%624, %602, %621, %arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512x256x4x16xf32>, %arg247: tensor<512x4x16x256xf32>, %arg248: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg248, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %643 = stablehlo.add %arg247, %642 : tensor<512x4x16x256xf32>
      %644 = stablehlo.transpose %643, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %645 = mhlo.copy %644 : tensor<512x256x4x16xf32>
      %646 = stablehlo.add %645, %arg246 : tensor<512x256x4x16xf32>
      %647 = mhlo.bitcast %646 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %648 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %649 = stablehlo.multiply %arg245, %648 : tensor<512xf32>
      %650 = stablehlo.broadcast_in_dim %649, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %651 = stablehlo.subtract %647, %650 : tensor<512x16384xf32>
      %652 = stablehlo.multiply %651, %651 : tensor<512x16384xf32>
      mhlo.return %652 : tensor<512x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>) -> tensor<512x16384xf32>
    %626 = "mhlo.fusion"(%625, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x16384xf32>, %arg246: tensor<f32>):
      %642 = "stablehlo.reduce_window"(%arg245, %arg246) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %643 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x16384xf32>, tensor<f32>) -> tensor<512x512xf32>
    %627 = "mhlo.fusion"(%626, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1] : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %642 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>, tensor<f32>) -> tensor<512xf32>
    %628 = "mhlo.fusion"(%627) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %644 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %645 = stablehlo.add %643, %644 : tensor<512xf32>
      %646 = stablehlo.rsqrt %645 : tensor<512xf32>
      mhlo.return %646 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %629 = "mhlo.fusion"(%arg187, %628, %624, %602, %621, %arg186, %arg188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x256x4x16xf32>, %arg249: tensor<512x4x16x256xf32>, %arg250: tensor<256xf32>, %arg251: tensor<256xf32>):
      %642 = stablehlo.broadcast_in_dim %arg251, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %643 = stablehlo.broadcast_in_dim %arg250, dims = [3] : (tensor<256xf32>) -> tensor<512x4x16x256xf32>
      %644 = stablehlo.add %arg249, %643 : tensor<512x4x16x256xf32>
      %645 = stablehlo.transpose %644, dims = [0, 3, 1, 2] {result_layout = dense<[1, 3, 2, 0]> : tensor<4xindex>, xla_shape = "f32[512,256,4,16]{1,3,2,0}"} : (tensor<512x4x16x256xf32>) -> tensor<512x256x4x16xf32>
      %646 = mhlo.copy %645 : tensor<512x256x4x16xf32>
      %647 = stablehlo.add %646, %arg248 : tensor<512x256x4x16xf32>
      %648 = mhlo.bitcast %647 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x16384xf32>
      %cst_1 = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %649 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %650 = stablehlo.multiply %arg247, %649 : tensor<512xf32>
      %651 = stablehlo.broadcast_in_dim %650, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %652 = stablehlo.subtract %648, %651 : tensor<512x16384xf32>
      %653 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<512xf32>) -> tensor<512x16384xf32>
      %654 = stablehlo.multiply %652, %653 : tensor<512x16384xf32>
      %655 = mhlo.bitcast %654 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x16384xf32>) -> tensor<512x256x4x16xf32>
      %656 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<512x256x4x16xf32>
      %657 = stablehlo.multiply %655, %656 : tensor<512x256x4x16xf32>
      %658 = stablehlo.add %642, %657 : tensor<512x256x4x16xf32>
      %659 = mhlo.bitcast %658 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x256x4x16xf32>) -> tensor<512x256x2x2x4x4xf32>
      %660 = stablehlo.transpose %659, dims = [2, 4, 3, 5, 0, 1] {result_layout = dense<[3, 1, 2, 0, 5, 4]> : tensor<6xindex>, xla_shape = "f32[2,4,2,4,512,256]{3,1,2,0,5,4}"} : (tensor<512x256x2x2x4x4xf32>) -> tensor<2x4x2x4x512x256xf32>
      %661 = mhlo.copy %660 : tensor<2x4x2x4x512x256xf32>
      %662 = mhlo.bitcast %661 {result_layout = dense<[2, 1, 0, 4, 3]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, xla_shape = "f32[64,512,256,1,1]{2,1,0,4,3}"} : (tensor<2x4x2x4x512x256xf32>) -> tensor<64x512x256x1x1xf32>
      %663 = mhlo.copy %662 : tensor<64x512x256x1x1xf32>
      mhlo.return %663 : tensor<64x512x256x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x4x16xf32>, tensor<512x4x16x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<64x512x256x1x1xf32>
    %630 = "mhlo.fusion"(%1, %2, %629) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x8x8xf32>, %arg246: tensor<64x2xi32>, %arg247: tensor<64x512x256x1x1xf32>):
      %642 = "stablehlo.scatter"(%arg245, %arg246, %arg247) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg248: tensor<f32>, %arg249: tensor<f32>):
        %643 = stablehlo.add %arg248, %arg249 : tensor<f32>
        stablehlo.return %643 : tensor<f32>
      }) : (tensor<512x256x8x8xf32>, tensor<64x2xi32>, tensor<64x512x256x1x1xf32>) -> tensor<512x256x8x8xf32>
      mhlo.return %642 : tensor<512x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x8x8xf32>, tensor<64x2xi32>, tensor<64x512x256x1x1xf32>) -> tensor<512x256x8x8xf32>
    %631 = "mhlo.fusion"(%630) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x8x8xf32>):
      %642 = stablehlo.transpose %arg245, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,8,8,256]{2,1,3,0}"} : (tensor<512x256x8x8xf32>) -> tensor<512x8x8x256xf32>
      %643 = mhlo.copy %642 : tensor<512x8x8x256xf32>
      mhlo.return %643 : tensor<512x8x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x8x8xf32>) -> tensor<512x8x8x256xf32>
    %632 = mhlo.bitcast %631 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x8x8x256xf32>) -> tensor<32768x256xf32>
    %633 = "mhlo.fusion"(%arg189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x256x1x1xf32>):
      %642 = stablehlo.transpose %arg245, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[1,1,256,512]{1,0,2,3}"} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
      %643 = mhlo.copy %642 : tensor<1x1x256x512xf32>
      mhlo.return %643 : tensor<1x1x256x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x256x1x1xf32>) -> tensor<1x1x256x512xf32>
    %634 = mhlo.bitcast %633 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %635 = stablehlo.dot %632, %634, precision = [DEFAULT, DEFAULT] : (tensor<32768x256xf32>, tensor<256x512xf32>) -> tensor<32768x512xf32>
    %636 = mhlo.bitcast %635 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<32768x512xf32>) -> tensor<512x8x8x512xf32>
    %637 = "mhlo.fusion"(%arg191, %arg190, %0, %636, %arg242) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<512x8x8x512xf32>, %arg249: tensor<512xf32>):
      %642 = stablehlo.broadcast_in_dim %arg249, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %643 = stablehlo.subtract %arg248, %642 : tensor<512x8x8x512xf32>
      %644 = stablehlo.broadcast_in_dim %arg247, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %645 = stablehlo.multiply %643, %644 : tensor<512x8x8x512xf32>
      %646 = stablehlo.broadcast_in_dim %arg246, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %647 = stablehlo.multiply %645, %646 : tensor<512x8x8x512xf32>
      %648 = stablehlo.broadcast_in_dim %arg245, dims = [3] : (tensor<512xf32>) -> tensor<512x8x8x512xf32>
      %649 = stablehlo.add %647, %648 : tensor<512x8x8x512xf32>
      mhlo.return %649 : tensor<512x8x8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x8x8x512xf32>, tensor<512xf32>) -> tensor<512x8x8x512xf32>
    %638 = "mhlo.fusion"(%637, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<512x8x8x512xf32>, %arg246: tensor<f32>):
      %642 = stablehlo.reduce(%arg245 init: %arg246) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x8x8x512xf32>, tensor<f32>) -> tensor<512x512xf32>
      mhlo.return %642 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x8x8x512xf32>, tensor<f32>) -> tensor<512x512xf32>
    %639 = "mhlo.fusion"(%638) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x512xf32>):
      %cst_1 = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %642 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x512xf32>
      %643 = stablehlo.multiply %arg245, %642 : tensor<512x512xf32>
      mhlo.return %643 : tensor<512x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %640 = stablehlo.dot_general %639, %arg192, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x512xf32>, tensor<2x512xf32>) -> tensor<512x2xf32>
    %641 = "mhlo.fusion"(%640, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512x2xf32>, %arg246: tensor<2xf32>):
      %642 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<2xf32>) -> tensor<512x2xf32>
      %643 = stablehlo.add %642, %arg245 : tensor<512x2xf32>
      mhlo.return %643 : tensor<512x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2xf32>, tensor<2xf32>) -> tensor<512x2xf32>
    return %641 : tensor<512x2xf32>
  }
}
