module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @region_66.75(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x768xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x768xf32>
    return %0 : tensor<512x768xf32>
  }
  func.func private @region_65.74(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.1(%arg0: tensor<512x1x1x3072xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
    return %0 : tensor<512x1x1x3xf32>
  }
  func.func private @region_57.64.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.2(%arg0: tensor<512x7x7x24xf32>, %arg1: tensor<f32>) -> tensor<512x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    return %0 : tensor<512x7x7xf32>
  }
  func.func private @region_63.72(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.3(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    return %0 : tensor<512x7x7xf32>
  }
  func.func private @region_62.71(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.4(%arg0: tensor<512x1x1x3072xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
    return %0 : tensor<512x1x1x3xf32>
  }
  func.func private @region_57.64.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.5(%arg0: tensor<512x7x7x24xf32>, %arg1: tensor<f32>) -> tensor<512x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    return %0 : tensor<512x7x7xf32>
  }
  func.func private @region_60.69(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.6(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    return %0 : tensor<512x7x7xf32>
  }
  func.func private @region_59.68(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.7(%arg0: tensor<512x1x1x3072xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
    return %0 : tensor<512x1x1x3xf32>
  }
  func.func private @region_57.64.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.8(%arg0: tensor<512x7x7x24xf32>, %arg1: tensor<f32>) -> tensor<512x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    return %0 : tensor<512x7x7xf32>
  }
  func.func private @region_55.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.9(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x7x7xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    return %0 : tensor<512x7x7xf32>
  }
  func.func private @region_27.32.clone.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.10(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_54.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.11(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_53.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.12(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.13(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_51.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.14(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_50.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.15(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.6.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.16(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_48.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.17(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_47.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.18(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.5.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.19(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_45.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.20(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_44.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.21(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.4.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.22(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_42.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.23(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_41.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.24(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.25(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_39.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.26(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_38.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.27(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.28(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_36.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.29(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_35.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.30(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.31(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_33.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.32(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_32.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.33(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.34(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_30.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.35(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_29.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.36(%arg0: tensor<512x1x1x1536xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x2xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    return %0 : tensor<512x1x1x2xf32>
  }
  func.func private @region_27.32.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.37(%arg0: tensor<512x14x14x12xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_25.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.38(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    return %0 : tensor<512x14x14xf32>
  }
  func.func private @region_15.18.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.39(%arg0: tensor<512x28x28x6xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_24.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.40(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_15.18.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.41(%arg0: tensor<512x28x28x6xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_21.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.42(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_15.18.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.43(%arg0: tensor<512x28x28x6xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_18.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.44(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_15.18.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.45(%arg0: tensor<512x28x28x6xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_13.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.46(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    return %0 : tensor<512x28x28xf32>
  }
  func.func private @region_2.3.clone.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.47(%arg0: tensor<512x56x56x3xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_12.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.48(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_10.13.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.49(%arg0: tensor<512x2x2x384xf32>, %arg1: tensor<f32>) -> tensor<512x384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
    return %0 : tensor<512x384xf32>
  }
  func.func private @region_2.3.clone.2.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.50(%arg0: tensor<512x56x56x3xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_9.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.51(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_7.10.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.52(%arg0: tensor<512x2x2x384xf32>, %arg1: tensor<f32>) -> tensor<512x384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
    return %0 : tensor<512x384xf32>
  }
  func.func private @region_2.3.clone.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.53(%arg0: tensor<512x56x56x3xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_6.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.54(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_4.7.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.55(%arg0: tensor<512x2x2x384xf32>, %arg1: tensor<f32>) -> tensor<512x384xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
    return %0 : tensor<512x384xf32>
  }
  func.func private @region_2.3.clone.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.56(%arg0: tensor<512x56x56x3xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_3.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.57(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_2.3.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.58(%arg0: tensor<512x56x56x3xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @region_0.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.59(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    return %0 : tensor<512x56x56xf32>
  }
  func.func private @fused_computation.60(%arg0: tensor<512x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<512x1000xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1000xf32>) -> tensor<512x1000xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<512x1000xf32>
    return %1 : tensor<512x1000xf32>
  }
  func.func private @fused_computation.61(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512x768xf32>) -> tensor<512x768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x768xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<512x768xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<512xf32>) -> tensor<512x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<512x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<512x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x768xf32>
    return %11 : tensor<512x768xf32>
  }
  func.func private @region_69.78.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.62(%arg0: tensor<512x24xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x24xf32>, tensor<f32>) -> tensor<512xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<512xf32>
    %cst_1 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %4 = stablehlo.add %2, %3 : tensor<512xf32>
    %5 = stablehlo.rsqrt %4 : tensor<512xf32>
    return %5 : tensor<512xf32>
  }
  func.func private @fused_computation.63(%arg0: tensor<512xf32>, %arg1: tensor<512x768xf32>) -> tensor<512x768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x768xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<512x768xf32>
    %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<512xf32>) -> tensor<512x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x768xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x768xf32>
    return %6 : tensor<512x768xf32>
  }
  func.func private @fused_computation.64(%arg0: tensor<512x768xf32>) -> tensor<512x768xf32> {
    %cst = stablehlo.constant dense<0.0204081628> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x768xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x768xf32>
    return %1 : tensor<512x768xf32>
  }
  func.func private @fused_computation.65(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<25088x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<25088x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<25088x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x768xf32>) -> tensor<512x7x7x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x7x7x768xf32>
    return %3 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.66(%arg0: tensor<512x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<512x1x1x3072xf32>, %arg3: tensor<1x1x1x3072xf32>) -> tensor<25088x3072xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x3072xf32>) -> tensor<512x3072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x3072xf32>) -> tensor<512x7x7x3072xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x7x7x3072xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x7x7x3072xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x7x7x3072xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x7x7x3072xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x3072xf32>) -> tensor<25088x3072xf32>
    return %10 : tensor<25088x3072xf32>
  }
  func.func private @fused_computation.67(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x3072xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x3072xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x3072xf32>
    return %8 : tensor<512x1x1x3072xf32>
  }
  func.func private @fused_computation.68(%arg0: tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x3072xf32>
    return %1 : tensor<512x1x1x3072xf32>
  }
  func.func private @region_64.73(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.69(%arg0: tensor<512x7x7x3072xf32>) -> tensor<512x3072xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x7x7x3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x3072xf32>, tensor<f32>) -> tensor<512x3072xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x3072xf32>
    return %2 : tensor<512x3072xf32>
  }
  func.func private @fused_computation.70(%arg0: tensor<25088x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x7x7x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<25088x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<25088x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<25088x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
    %5 = stablehlo.negate %2 : tensor<25088x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<25088x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x7x7x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x7x7x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x7x7x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x7x7x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x7x7x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x7x7x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x7x7x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x7x7x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x7x7x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x7x7x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x7x7x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x7x7x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x7x7x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x7x7x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x7x7x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x7x7x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x7x7x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x7x7x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x7x7x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x7x7x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x7x7x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x7x7x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x7x7x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x7x7x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x7x7x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x7x7x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x7x7x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x7x7x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x7x7x3072xf32>
    return %98 : tensor<512x7x7x3072xf32>
  }
  func.func private @fused_computation.71(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x7x7xf32>, %arg3: tensor<512x7x7xf32>, %arg4: tensor<512x7x7x768xf32>, %arg5: tensor<768xf32>) -> tensor<25088x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x7x7x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x7x7x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x7x7x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x7x7x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x768xf32>) -> tensor<25088x768xf32>
    return %12 : tensor<25088x768xf32>
  }
  func.func private @fused_computation.72(%arg0: tensor<512x7x7xf32>) -> tensor<512x7x7xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x7x7xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x7x7xf32>
    return %4 : tensor<512x7x7xf32>
  }
  func.func private @fused_computation.73(%arg0: tensor<512x7x7xf32>, %arg1: tensor<512x7x7x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x7x7x768xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x7x7x768xf32>
    return %6 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.74(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x7x7x768xf32>
    return %1 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.75(%arg0: tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x768xf32>
    return %1 : tensor<7x7x1x768xf32>
  }
  func.func private @fused_computation.76(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<25088x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<25088x768xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<25088x768xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x768xf32>) -> tensor<512x7x7x768xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x7x7x768xf32>
    return %3 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.77(%arg0: tensor<512x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<512x1x1x3072xf32>, %arg3: tensor<1x1x1x3072xf32>) -> tensor<25088x3072xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x3072xf32>) -> tensor<512x3072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x3072xf32>) -> tensor<512x7x7x3072xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x7x7x3072xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x7x7x3072xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x7x7x3072xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x7x7x3072xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x3072xf32>) -> tensor<25088x3072xf32>
    return %10 : tensor<25088x3072xf32>
  }
  func.func private @fused_computation.78(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x3072xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x3072xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x3072xf32>
    return %8 : tensor<512x1x1x3072xf32>
  }
  func.func private @fused_computation.79(%arg0: tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x3072xf32>
    return %1 : tensor<512x1x1x3072xf32>
  }
  func.func private @region_61.70(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.80(%arg0: tensor<512x7x7x3072xf32>) -> tensor<512x3072xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x7x7x3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x3072xf32>, tensor<f32>) -> tensor<512x3072xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x3072xf32>
    return %2 : tensor<512x3072xf32>
  }
  func.func private @fused_computation.81(%arg0: tensor<25088x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x7x7x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<25088x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<25088x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<25088x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
    %5 = stablehlo.negate %2 : tensor<25088x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<25088x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x7x7x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x7x7x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x7x7x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x7x7x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x7x7x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x7x7x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x7x7x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x7x7x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x7x7x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x7x7x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x7x7x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x7x7x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x7x7x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x7x7x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x7x7x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x7x7x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x7x7x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x7x7x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x7x7x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x7x7x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x7x7x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x7x7x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x7x7x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x7x7x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x7x7x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x7x7x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x7x7x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x7x7x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x7x7x3072xf32>
    return %98 : tensor<512x7x7x3072xf32>
  }
  func.func private @fused_computation.82(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x7x7xf32>, %arg3: tensor<512x7x7xf32>, %arg4: tensor<512x7x7x768xf32>, %arg5: tensor<768xf32>) -> tensor<25088x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x7x7x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x7x7x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x7x7x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x7x7x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x768xf32>) -> tensor<25088x768xf32>
    return %12 : tensor<25088x768xf32>
  }
  func.func private @fused_computation.83(%arg0: tensor<512x7x7xf32>) -> tensor<512x7x7xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x7x7xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x7x7xf32>
    return %4 : tensor<512x7x7xf32>
  }
  func.func private @fused_computation.84(%arg0: tensor<512x7x7xf32>, %arg1: tensor<512x7x7x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x7x7x768xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x7x7x768xf32>
    return %6 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.85(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x7x7x768xf32>
    return %1 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.86(%arg0: tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x768xf32>
    return %1 : tensor<7x7x1x768xf32>
  }
  func.func private @fused_computation.87(%arg0: tensor<25088x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x7x7x768xf32>, %arg3: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<512x7x7x768xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<25088x768xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<25088x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x768xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.add %1, %4 : tensor<512x7x7x768xf32>
    return %5 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.88(%arg0: tensor<512x7x7x3072xf32>, %arg1: tensor<1x1x1x3072xf32>, %arg2: tensor<512x1x1x3072xf32>, %arg3: tensor<1x1x1x3072xf32>) -> tensor<25088x3072xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x3072xf32>) -> tensor<512x3072xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x3072xf32>) -> tensor<512x7x7x3072xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x7x7x3072xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x7x7x3072xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x7x7x3072xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x7x7x3072xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x3072xf32>) -> tensor<25088x3072xf32>
    return %10 : tensor<25088x3072xf32>
  }
  func.func private @fused_computation.89(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x3072xf32>
    %cst = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x3072xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x3072xf32>
    return %8 : tensor<512x1x1x3072xf32>
  }
  func.func private @fused_computation.90(%arg0: tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x3072xf32>
    return %1 : tensor<512x1x1x3072xf32>
  }
  func.func private @region_58.67(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.91(%arg0: tensor<512x7x7x3072xf32>) -> tensor<512x3072xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x7x7x3072xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x3072xf32>, tensor<f32>) -> tensor<512x3072xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x3072xf32>
    return %2 : tensor<512x3072xf32>
  }
  func.func private @fused_computation.92(%arg0: tensor<25088x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<512x7x7x3072xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3072xf32>) -> tensor<25088x3072xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<25088x3072xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<25088x3072xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
    %5 = stablehlo.negate %2 : tensor<25088x3072xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<25088x3072xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
    %9 = stablehlo.abs %8 : tensor<512x7x7x3072xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x7x7x3072xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x7x7x3072xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x7x7x3072xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x7x7x3072xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x7x7x3072xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x7x7x3072xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x7x7x3072xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x7x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x7x7x3072xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x7x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x7x7x3072xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x7x7x3072xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x7x7x3072xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x7x7x3072xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x7x7x3072xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %37 = stablehlo.negate %13 : tensor<512x7x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %40 = stablehlo.exponential %37 : tensor<512x7x7x3072xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x7x7x3072xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x7x7x3072xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x7x7x3072xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x7x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x7x7x3072xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x7x7x3072xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x7x7x3072xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x7x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x7x7x3072xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x7x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x7x7x3072xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x7x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x7x7x3072xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x7x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x7x7x3072xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x7x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x7x7x3072xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x7x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x7x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x7x7x3072xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x7x7x3072xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x7x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x7x7x3072xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x7x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x7x7x3072xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x7x7x3072xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x7x7x3072xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x7x7x3072xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x7x7x3072xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x7x7x3072xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x7x7x3072xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x7x7x3072xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x7x7x3072xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x7x7x3072xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x7x7x3072xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x7x7x3072xf32>
    return %98 : tensor<512x7x7x3072xf32>
  }
  func.func private @fused_computation.93(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<512x7x7xf32>, %arg3: tensor<512x7x7xf32>, %arg4: tensor<512x7x7x768xf32>, %arg5: tensor<768xf32>) -> tensor<25088x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x7x7x768xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x7x7x768xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x7x7x768xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x7x7x768xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x768xf32>) -> tensor<25088x768xf32>
    return %12 : tensor<25088x768xf32>
  }
  func.func private @fused_computation.94(%arg0: tensor<512x7x7xf32>) -> tensor<512x7x7xf32> {
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x7x7xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x7x7xf32>
    return %4 : tensor<512x7x7xf32>
  }
  func.func private @fused_computation.95(%arg0: tensor<512x7x7xf32>, %arg1: tensor<512x7x7x768xf32>, %arg2: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x7x7x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x7x7xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x7x7x768xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x7x7x768xf32>
    return %6 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.96(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x7x7x768xf32>
    return %1 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.97(%arg0: tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x768xf32>
    return %1 : tensor<7x7x1x768xf32>
  }
  func.func private @fused_computation.98(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x7x7x768xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x7x7x768xf32>
    return %1 : tensor<512x7x7x768xf32>
  }
  func.func private @fused_computation.99(%arg0: tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,384,768]{1,0,2,3}"} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
    %1 = mhlo.copy %0 : tensor<2x2x384x768xf32>
    return %1 : tensor<2x2x384x768xf32>
  }
  func.func private @fused_computation.100(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14x384xf32>, %arg4: tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<512x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<512x14x14x384xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x14x14x384xf32>
    return %9 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.101(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.102(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<512x14x14xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<512x14x14x384xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<512x14x14x384xf32>
    return %4 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.103(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.104(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.105(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.106(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_52.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.107(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.108(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.109(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.110(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.111(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.112(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.113(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.114(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.115(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.116(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.117(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_49.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.118(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.119(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.120(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.121(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.122(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.123(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.124(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.125(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.126(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.127(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.128(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_46.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.129(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.130(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.131(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.132(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.133(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.134(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.135(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.136(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.137(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.138(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.139(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_43.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.140(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.141(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.142(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.143(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.144(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.145(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.146(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.147(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.148(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.149(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.150(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_40.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.151(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.152(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.153(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.154(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.155(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.156(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.157(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.158(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.159(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.160(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.161(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_37.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.162(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.163(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.164(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.165(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.166(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.167(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.168(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.169(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.170(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.171(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.172(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_34.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.173(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.174(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.175(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.176(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.177(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.178(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.179(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.180(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<100352x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<100352x384xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x14x14x384xf32>
    return %3 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.181(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.182(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.183(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_31.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.184(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.185(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.186(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.187(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.188(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.189(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.190(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.191(%arg0: tensor<100352x384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14x384xf32>, %arg3: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<512x14x14x384xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<100352x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.add %1, %4 : tensor<512x14x14x384xf32>
    return %5 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.192(%arg0: tensor<512x14x14x1536xf32>, %arg1: tensor<1x1x1x1536xf32>, %arg2: tensor<512x1x1x1536xf32>, %arg3: tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x14x14x1536xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x14x14x1536xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x14x14x1536xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x14x14x1536xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
    return %10 : tensor<100352x1536xf32>
  }
  func.func private @fused_computation.193(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    %cst = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %5 = stablehlo.add %3, %4 : tensor<512x1x1xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
    %8 = stablehlo.divide %1, %7 : tensor<512x1x1x1536xf32>
    return %8 : tensor<512x1x1x1536xf32>
  }
  func.func private @fused_computation.194(%arg0: tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %1 = mhlo.copy %0 : tensor<512x1x1x1536xf32>
    return %1 : tensor<512x1x1x1536xf32>
  }
  func.func private @region_28.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.195(%arg0: tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x14x14x1536xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x1536xf32>
    return %2 : tensor<512x1536xf32>
  }
  func.func private @fused_computation.196(%arg0: tensor<100352x1536xf32>, %arg1: tensor<1536xf32>) -> tensor<512x14x14x1536xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<100352x1536xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<100352x1536xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %5 = stablehlo.negate %2 : tensor<100352x1536xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<100352x1536xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
    %9 = stablehlo.abs %8 : tensor<512x14x14x1536xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x14x14x1536xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x14x14x1536xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x14x14x1536xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x14x14x1536xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x14x14x1536xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x14x14x1536xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x14x14x1536xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x14x14x1536xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x14x14x1536xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x14x14x1536xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x14x14x1536xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x14x14x1536xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x14x14x1536xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x14x14x1536xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x14x14x1536xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %37 = stablehlo.negate %13 : tensor<512x14x14x1536xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %40 = stablehlo.exponential %37 : tensor<512x14x14x1536xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x14x14x1536xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x14x14x1536xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x14x14x1536xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x14x14x1536xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x14x14x1536xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x14x14x1536xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x14x14x1536xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x14x14x1536xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x14x14x1536xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x14x14x1536xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x14x14x1536xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x14x14x1536xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x14x14x1536xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x14x14x1536xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x14x14x1536xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x14x14x1536xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x14x14x1536xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x14x14x1536xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x14x14x1536xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x14x14x1536xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x14x14x1536xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x14x14x1536xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x14x14x1536xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x14x14x1536xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x14x14x1536xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x14x14x1536xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x14x14x1536xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x14x14x1536xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x14x14x1536xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x14x14x1536xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x14x14x1536xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x14x14x1536xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x14x14x1536xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x14x14x1536xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x14x14x1536xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x14x14x1536xf32>
    return %98 : tensor<512x14x14x1536xf32>
  }
  func.func private @fused_computation.197(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<512x14x14xf32>, %arg3: tensor<512x14x14xf32>, %arg4: tensor<512x14x14x384xf32>, %arg5: tensor<384xf32>) -> tensor<100352x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x14x14x384xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x14x14x384xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x14x14x384xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
    return %12 : tensor<100352x384xf32>
  }
  func.func private @fused_computation.198(%arg0: tensor<512x14x14xf32>) -> tensor<512x14x14xf32> {
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x14x14xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x14x14xf32>
    return %4 : tensor<512x14x14xf32>
  }
  func.func private @fused_computation.199(%arg0: tensor<512x14x14xf32>, %arg1: tensor<512x14x14x384xf32>, %arg2: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x14x14x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x14x14xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x14x14x384xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x14x14x384xf32>
    return %6 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.200(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.201(%arg0: tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x384xf32>
    return %1 : tensor<7x7x1x384xf32>
  }
  func.func private @fused_computation.202(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x14x14x384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x14x14x384xf32>
    return %1 : tensor<512x14x14x384xf32>
  }
  func.func private @fused_computation.203(%arg0: tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,192,384]{1,0,2,3}"} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
    %1 = mhlo.copy %0 : tensor<2x2x192x384xf32>
    return %1 : tensor<2x2x192x384xf32>
  }
  func.func private @fused_computation.204(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<512x28x28xf32>, %arg3: tensor<512x28x28x192xf32>, %arg4: tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<512x28x28x192xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x28x28x192xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x28x28x192xf32>
    return %9 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.205(%arg0: tensor<512x28x28xf32>) -> tensor<512x28x28xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x28x28xf32>
    return %4 : tensor<512x28x28xf32>
  }
  func.func private @fused_computation.206(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<512x28x28xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<512x28x28x192xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<512x28x28x192xf32>
    return %4 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.207(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<401408x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<401408x192xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<401408x192xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x192xf32>) -> tensor<512x28x28x192xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x28x28x192xf32>
    return %3 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.208(%arg0: tensor<512x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<512x1x1x768xf32>, %arg3: tensor<1x1x1x768xf32>) -> tensor<401408x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x768xf32>) -> tensor<512x768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x768xf32>) -> tensor<512x28x28x768xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x28x28x768xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x28x28x768xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x28x28x768xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x28x28x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x768xf32>) -> tensor<401408x768xf32>
    return %10 : tensor<401408x768xf32>
  }
  func.func private @fused_computation.209(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x768xf32>) -> tensor<512x1x1x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<512x1x1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x768xf32>
    %7 = stablehlo.divide %0, %6 : tensor<512x1x1x768xf32>
    return %7 : tensor<512x1x1x768xf32>
  }
  func.func private @region_22.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.210(%arg0: tensor<512x28x28x768xf32>) -> tensor<512x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x28x28x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x28x28x768xf32>, tensor<f32>) -> tensor<512x768xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x768xf32>
    return %2 : tensor<512x768xf32>
  }
  func.func private @fused_computation.211(%arg0: tensor<401408x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x28x28x768xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<401408x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<401408x768xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<401408x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
    %5 = stablehlo.negate %2 : tensor<401408x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<401408x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
    %9 = stablehlo.abs %8 : tensor<512x28x28x768xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x28x28x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x28x28x768xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x28x28x768xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x28x28x768xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x28x28x768xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x28x28x768xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x28x28x768xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x28x28x768xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x28x28x768xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x28x28x768xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x28x28x768xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x28x28x768xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x28x28x768xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x28x28x768xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x28x28x768xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %37 = stablehlo.negate %13 : tensor<512x28x28x768xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %40 = stablehlo.exponential %37 : tensor<512x28x28x768xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x28x28x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x28x28x768xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x28x28x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x28x28x768xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x28x28x768xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x28x28x768xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x28x28x768xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x28x28x768xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x28x28x768xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x28x28x768xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x28x28x768xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x28x28x768xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x28x28x768xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x28x28x768xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x28x28x768xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x28x28x768xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x28x28x768xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x28x28x768xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x28x28x768xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x28x28x768xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x28x28x768xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x28x28x768xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x28x28x768xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x28x28x768xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x28x28x768xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x28x28x768xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x28x28x768xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x28x28x768xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x28x28x768xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x28x28x768xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x28x28x768xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x28x28x768xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x28x28x768xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x28x28x768xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x28x28x768xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x28x28x768xf32>
    return %98 : tensor<512x28x28x768xf32>
  }
  func.func private @fused_computation.212(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<512x28x28xf32>, %arg3: tensor<512x28x28xf32>, %arg4: tensor<512x28x28x192xf32>, %arg5: tensor<192xf32>) -> tensor<401408x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x28x28x192xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x28x28x192xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x192xf32>) -> tensor<401408x192xf32>
    return %12 : tensor<401408x192xf32>
  }
  func.func private @fused_computation.213(%arg0: tensor<512x28x28xf32>) -> tensor<512x28x28xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x28x28xf32>
    return %4 : tensor<512x28x28xf32>
  }
  func.func private @fused_computation.214(%arg0: tensor<512x28x28xf32>, %arg1: tensor<512x28x28x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x28x28x192xf32>
    return %6 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.215(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x28x28x192xf32>
    return %1 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.216(%arg0: tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x192xf32>
    return %1 : tensor<7x7x1x192xf32>
  }
  func.func private @fused_computation.217(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<401408x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<401408x192xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<401408x192xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x192xf32>) -> tensor<512x28x28x192xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x28x28x192xf32>
    return %3 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.218(%arg0: tensor<512x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<512x1x1x768xf32>, %arg3: tensor<1x1x1x768xf32>) -> tensor<401408x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x768xf32>) -> tensor<512x768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x768xf32>) -> tensor<512x28x28x768xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x28x28x768xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x28x28x768xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x28x28x768xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x28x28x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x768xf32>) -> tensor<401408x768xf32>
    return %10 : tensor<401408x768xf32>
  }
  func.func private @fused_computation.219(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x768xf32>) -> tensor<512x1x1x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<512x1x1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x768xf32>
    %7 = stablehlo.divide %0, %6 : tensor<512x1x1x768xf32>
    return %7 : tensor<512x1x1x768xf32>
  }
  func.func private @region_19.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.220(%arg0: tensor<512x28x28x768xf32>) -> tensor<512x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x28x28x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x28x28x768xf32>, tensor<f32>) -> tensor<512x768xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x768xf32>
    return %2 : tensor<512x768xf32>
  }
  func.func private @fused_computation.221(%arg0: tensor<401408x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x28x28x768xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<401408x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<401408x768xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<401408x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
    %5 = stablehlo.negate %2 : tensor<401408x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<401408x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
    %9 = stablehlo.abs %8 : tensor<512x28x28x768xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x28x28x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x28x28x768xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x28x28x768xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x28x28x768xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x28x28x768xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x28x28x768xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x28x28x768xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x28x28x768xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x28x28x768xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x28x28x768xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x28x28x768xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x28x28x768xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x28x28x768xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x28x28x768xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x28x28x768xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %37 = stablehlo.negate %13 : tensor<512x28x28x768xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %40 = stablehlo.exponential %37 : tensor<512x28x28x768xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x28x28x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x28x28x768xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x28x28x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x28x28x768xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x28x28x768xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x28x28x768xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x28x28x768xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x28x28x768xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x28x28x768xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x28x28x768xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x28x28x768xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x28x28x768xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x28x28x768xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x28x28x768xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x28x28x768xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x28x28x768xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x28x28x768xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x28x28x768xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x28x28x768xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x28x28x768xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x28x28x768xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x28x28x768xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x28x28x768xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x28x28x768xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x28x28x768xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x28x28x768xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x28x28x768xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x28x28x768xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x28x28x768xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x28x28x768xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x28x28x768xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x28x28x768xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x28x28x768xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x28x28x768xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x28x28x768xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x28x28x768xf32>
    return %98 : tensor<512x28x28x768xf32>
  }
  func.func private @fused_computation.222(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<512x28x28xf32>, %arg3: tensor<512x28x28xf32>, %arg4: tensor<512x28x28x192xf32>, %arg5: tensor<192xf32>) -> tensor<401408x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x28x28x192xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x28x28x192xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x192xf32>) -> tensor<401408x192xf32>
    return %12 : tensor<401408x192xf32>
  }
  func.func private @fused_computation.223(%arg0: tensor<512x28x28xf32>) -> tensor<512x28x28xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x28x28xf32>
    return %4 : tensor<512x28x28xf32>
  }
  func.func private @fused_computation.224(%arg0: tensor<512x28x28xf32>, %arg1: tensor<512x28x28x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x28x28x192xf32>
    return %6 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.225(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x28x28x192xf32>
    return %1 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.226(%arg0: tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x192xf32>
    return %1 : tensor<7x7x1x192xf32>
  }
  func.func private @fused_computation.227(%arg0: tensor<401408x192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<512x28x28x192xf32>, %arg3: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg2, %0 : tensor<512x28x28x192xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<401408x192xf32>
    %3 = stablehlo.add %2, %arg0 : tensor<401408x192xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x192xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.add %1, %4 : tensor<512x28x28x192xf32>
    return %5 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.228(%arg0: tensor<512x28x28x768xf32>, %arg1: tensor<1x1x1x768xf32>, %arg2: tensor<512x1x1x768xf32>, %arg3: tensor<1x1x1x768xf32>) -> tensor<401408x768xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x768xf32>) -> tensor<512x768xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x768xf32>) -> tensor<512x28x28x768xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x28x28x768xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x28x28x768xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x28x28x768xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x28x28x768xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x768xf32>) -> tensor<401408x768xf32>
    return %10 : tensor<401408x768xf32>
  }
  func.func private @fused_computation.229(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x768xf32>) -> tensor<512x1x1x768xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %cst = stablehlo.constant dense<0.00130208337> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<512x1x1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x768xf32>
    %7 = stablehlo.divide %0, %6 : tensor<512x1x1x768xf32>
    return %7 : tensor<512x1x1x768xf32>
  }
  func.func private @region_16.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.230(%arg0: tensor<512x28x28x768xf32>) -> tensor<512x768xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x28x28x768xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x28x28x768xf32>, tensor<f32>) -> tensor<512x768xf32>
    %2 = stablehlo.sqrt %1 : tensor<512x768xf32>
    return %2 : tensor<512x768xf32>
  }
  func.func private @fused_computation.231(%arg0: tensor<401408x768xf32>, %arg1: tensor<768xf32>) -> tensor<512x28x28x768xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<401408x768xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<401408x768xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<401408x768xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
    %5 = stablehlo.negate %2 : tensor<401408x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<401408x768xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
    %9 = stablehlo.abs %8 : tensor<512x28x28x768xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x28x28x768xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x28x28x768xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x28x28x768xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x28x28x768xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x28x28x768xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x28x28x768xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x28x28x768xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x28x28x768xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x28x28x768xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x28x28x768xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x28x28x768xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x28x28x768xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x28x28x768xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x28x28x768xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x28x28x768xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %37 = stablehlo.negate %13 : tensor<512x28x28x768xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %40 = stablehlo.exponential %37 : tensor<512x28x28x768xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x28x28x768xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x28x28x768xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x28x28x768xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x28x28x768xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x28x28x768xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x28x28x768xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x28x28x768xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x28x28x768xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x28x28x768xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x28x28x768xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x28x28x768xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x28x28x768xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x28x28x768xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x28x28x768xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x28x28x768xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x28x28x768xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x28x28x768xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x28x28x768xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x28x28x768xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x28x28x768xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x28x28x768xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x28x28x768xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x28x28x768xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x28x28x768xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x28x28x768xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x28x28x768xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x28x28x768xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x28x28x768xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x28x28x768xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x28x28x768xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x28x28x768xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x28x28x768xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x28x28x768xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x28x28x768xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x28x28x768xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x28x28x768xf32>
    return %98 : tensor<512x28x28x768xf32>
  }
  func.func private @fused_computation.232(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<512x28x28xf32>, %arg3: tensor<512x28x28xf32>, %arg4: tensor<512x28x28x192xf32>, %arg5: tensor<192xf32>) -> tensor<401408x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x28x28x192xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x28x28x192xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x28x28x192xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x192xf32>) -> tensor<401408x192xf32>
    return %12 : tensor<401408x192xf32>
  }
  func.func private @fused_computation.233(%arg0: tensor<512x28x28xf32>) -> tensor<512x28x28xf32> {
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x28x28xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x28x28xf32>
    return %4 : tensor<512x28x28xf32>
  }
  func.func private @fused_computation.234(%arg0: tensor<512x28x28xf32>, %arg1: tensor<512x28x28x192xf32>, %arg2: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x28x28x192xf32>
    %cst = stablehlo.constant dense<0.00520833349> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x28x28xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x28x28x192xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x28x28x192xf32>
    return %6 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.235(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x28x28x192xf32>
    return %1 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.236(%arg0: tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x192xf32>
    return %1 : tensor<7x7x1x192xf32>
  }
  func.func private @fused_computation.237(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<192xf32>) -> tensor<512x28x28x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x28x28x192xf32>
    return %1 : tensor<512x28x28x192xf32>
  }
  func.func private @fused_computation.238(%arg0: tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,96,192]{1,0,2,3}"} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
    %1 = mhlo.copy %0 : tensor<2x2x96x192xf32>
    return %1 : tensor<2x2x96x192xf32>
  }
  func.func private @fused_computation.239(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<512x56x56xf32>, %arg3: tensor<512x56x56x96xf32>, %arg4: tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg4, %0 : tensor<512x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<512x56x56x96xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %9 = stablehlo.add %7, %8 : tensor<512x56x56x96xf32>
    return %9 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.240(%arg0: tensor<512x56x56xf32>) -> tensor<512x56x56xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x56x56xf32>
    return %4 : tensor<512x56x56xf32>
  }
  func.func private @fused_computation.241(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<512x56x56xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<512x56x56x96xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<512x56x56x96xf32>
    return %4 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.242(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<1605632x96xf32>, %arg2: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<96xf32>) -> tensor<1605632x96xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1605632x96xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x96xf32>) -> tensor<512x56x56x96xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x56x56x96xf32>
    return %3 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.243(%arg0: tensor<512x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<512x1x1x384xf32>, %arg3: tensor<1x1x1x384xf32>) -> tensor<1605632x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x384xf32>) -> tensor<512x384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x384xf32>) -> tensor<512x56x56x384xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x56x56x384xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x56x56x384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x56x56x384xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x56x56x384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x384xf32>) -> tensor<1605632x384xf32>
    return %10 : tensor<1605632x384xf32>
  }
  func.func private @fused_computation.244(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x384xf32>) -> tensor<512x1x1x384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<512x1x1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x384xf32>
    %7 = stablehlo.divide %0, %6 : tensor<512x1x1x384xf32>
    return %7 : tensor<512x1x1x384xf32>
  }
  func.func private @fused_computation.245(%arg0: tensor<1605632x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x56x56x384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1605632x384xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<1605632x384xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1605632x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
    %5 = stablehlo.negate %2 : tensor<1605632x384xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1605632x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
    %9 = stablehlo.abs %8 : tensor<512x56x56x384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x56x56x384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x56x56x384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x56x56x384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x56x56x384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x56x56x384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x56x56x384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x56x56x384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x56x56x384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x56x56x384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x56x56x384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x56x56x384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x56x56x384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x56x56x384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x56x56x384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x56x56x384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %37 = stablehlo.negate %13 : tensor<512x56x56x384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %40 = stablehlo.exponential %37 : tensor<512x56x56x384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x56x56x384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x56x56x384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x56x56x384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x56x56x384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x56x56x384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x56x56x384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x56x56x384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x56x56x384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x56x56x384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x56x56x384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x56x56x384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x56x56x384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x56x56x384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x56x56x384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x56x56x384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x56x56x384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x56x56x384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x56x56x384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x56x56x384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x56x56x384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x56x56x384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x56x56x384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x56x56x384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x56x56x384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x56x56x384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x56x56x384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x56x56x384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x56x56x384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x56x56x384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x56x56x384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x56x56x384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x56x56x384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x56x56x384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x56x56x384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x56x56x384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x56x56x384xf32>
    return %98 : tensor<512x56x56x384xf32>
  }
  func.func private @fused_computation.246(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<512x56x56xf32>, %arg3: tensor<512x56x56xf32>, %arg4: tensor<512x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<1605632x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x56x56x96xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x96xf32>) -> tensor<1605632x96xf32>
    return %12 : tensor<1605632x96xf32>
  }
  func.func private @fused_computation.247(%arg0: tensor<512x56x56xf32>) -> tensor<512x56x56xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x56x56xf32>
    return %4 : tensor<512x56x56xf32>
  }
  func.func private @fused_computation.248(%arg0: tensor<512x56x56xf32>, %arg1: tensor<512x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x56x56x96xf32>
    return %6 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.249(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x56x56x96xf32>
    return %1 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.250(%arg0: tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x96xf32>
    return %1 : tensor<7x7x1x96xf32>
  }
  func.func private @fused_computation.251(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<1605632x96xf32>, %arg2: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<96xf32>) -> tensor<1605632x96xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1605632x96xf32>
    %2 = mhlo.bitcast %1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x96xf32>) -> tensor<512x56x56x96xf32>
    %3 = stablehlo.add %arg0, %2 : tensor<512x56x56x96xf32>
    return %3 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.252(%arg0: tensor<512x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<512x1x1x384xf32>, %arg3: tensor<1x1x1x384xf32>) -> tensor<1605632x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x384xf32>) -> tensor<512x384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x384xf32>) -> tensor<512x56x56x384xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x56x56x384xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x56x56x384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x56x56x384xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x56x56x384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x384xf32>) -> tensor<1605632x384xf32>
    return %10 : tensor<1605632x384xf32>
  }
  func.func private @fused_computation.253(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x384xf32>) -> tensor<512x1x1x384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<512x1x1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x384xf32>
    %7 = stablehlo.divide %0, %6 : tensor<512x1x1x384xf32>
    return %7 : tensor<512x1x1x384xf32>
  }
  func.func private @fused_computation.254(%arg0: tensor<1605632x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x56x56x384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1605632x384xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<1605632x384xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1605632x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
    %5 = stablehlo.negate %2 : tensor<1605632x384xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1605632x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
    %9 = stablehlo.abs %8 : tensor<512x56x56x384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x56x56x384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x56x56x384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x56x56x384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x56x56x384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x56x56x384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x56x56x384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x56x56x384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x56x56x384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x56x56x384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x56x56x384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x56x56x384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x56x56x384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x56x56x384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x56x56x384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x56x56x384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %37 = stablehlo.negate %13 : tensor<512x56x56x384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %40 = stablehlo.exponential %37 : tensor<512x56x56x384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x56x56x384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x56x56x384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x56x56x384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x56x56x384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x56x56x384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x56x56x384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x56x56x384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x56x56x384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x56x56x384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x56x56x384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x56x56x384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x56x56x384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x56x56x384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x56x56x384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x56x56x384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x56x56x384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x56x56x384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x56x56x384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x56x56x384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x56x56x384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x56x56x384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x56x56x384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x56x56x384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x56x56x384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x56x56x384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x56x56x384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x56x56x384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x56x56x384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x56x56x384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x56x56x384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x56x56x384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x56x56x384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x56x56x384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x56x56x384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x56x56x384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x56x56x384xf32>
    return %98 : tensor<512x56x56x384xf32>
  }
  func.func private @fused_computation.255(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<512x56x56xf32>, %arg3: tensor<512x56x56xf32>, %arg4: tensor<512x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<1605632x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x56x56x96xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x96xf32>) -> tensor<1605632x96xf32>
    return %12 : tensor<1605632x96xf32>
  }
  func.func private @fused_computation.256(%arg0: tensor<512x56x56xf32>) -> tensor<512x56x56xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x56x56xf32>
    return %4 : tensor<512x56x56xf32>
  }
  func.func private @fused_computation.257(%arg0: tensor<512x56x56xf32>, %arg1: tensor<512x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x56x56x96xf32>
    return %6 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.258(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x56x56x96xf32>
    return %1 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.259(%arg0: tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x96xf32>
    return %1 : tensor<7x7x1x96xf32>
  }
  func.func private @fused_computation.260(%arg0: tensor<1605632x96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<96xf32>, %arg4: tensor<512x56x56xf32>, %arg5: tensor<512x56x56xf32>, %arg6: tensor<512x56x56x96xf32>, %arg7: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg7, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg6, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg5, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.broadcast_in_dim %arg4, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x56x56x96xf32>
    %12 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<96xf32>) -> tensor<1605632x96xf32>
    %13 = stablehlo.add %12, %arg0 : tensor<1605632x96xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x96xf32>) -> tensor<512x56x56x96xf32>
    %15 = stablehlo.add %11, %14 : tensor<512x56x56x96xf32>
    return %15 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.261(%arg0: tensor<512x56x56x384xf32>, %arg1: tensor<1x1x1x384xf32>, %arg2: tensor<512x1x1x384xf32>, %arg3: tensor<1x1x1x384xf32>) -> tensor<1605632x384xf32> {
    %0 = mhlo.bitcast %arg3 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %2 = mhlo.bitcast %arg2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x384xf32>) -> tensor<512x384xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 3] : (tensor<512x384xf32>) -> tensor<512x56x56x384xf32>
    %4 = stablehlo.multiply %arg0, %3 : tensor<512x56x56x384xf32>
    %5 = stablehlo.multiply %1, %4 : tensor<512x56x56x384xf32>
    %6 = mhlo.bitcast %arg1 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %8 = stablehlo.add %5, %7 : tensor<512x56x56x384xf32>
    %9 = stablehlo.add %8, %arg0 : tensor<512x56x56x384xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x384xf32>) -> tensor<1605632x384xf32>
    return %10 : tensor<1605632x384xf32>
  }
  func.func private @fused_computation.262(%arg0: tensor<512x1x1xf32>, %arg1: tensor<512x384xf32>) -> tensor<512x1x1x384xf32> {
    %0 = mhlo.bitcast %arg1 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %cst = stablehlo.constant dense<0.00260416674> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<512x1x1xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<512x1x1xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x384xf32>
    %7 = stablehlo.divide %0, %6 : tensor<512x1x1x384xf32>
    return %7 : tensor<512x1x1x384xf32>
  }
  func.func private @fused_computation.263(%arg0: tensor<1605632x384xf32>, %arg1: tensor<384xf32>) -> tensor<512x56x56x384xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1605632x384xf32>
    %2 = stablehlo.add %1, %arg0 : tensor<1605632x384xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<1605632x384xf32>
    %4 = mhlo.bitcast %3 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
    %5 = stablehlo.negate %2 : tensor<1605632x384xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1605632x384xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
    %9 = stablehlo.abs %8 : tensor<512x56x56x384xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %11 = stablehlo.compare LT, %9, %10 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_2 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %13 = stablehlo.multiply %8, %8 : tensor<512x56x56x384xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<512x56x56x384xf32>
    %cst_3 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %16 = stablehlo.add %14, %15 : tensor<512x56x56x384xf32>
    %17 = stablehlo.multiply %16, %13 : tensor<512x56x56x384xf32>
    %cst_4 = stablehlo.constant dense<0.00518832775> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %19 = stablehlo.add %17, %18 : tensor<512x56x56x384xf32>
    %20 = stablehlo.multiply %19, %13 : tensor<512x56x56x384xf32>
    %cst_5 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %22 = stablehlo.add %20, %21 : tensor<512x56x56x384xf32>
    %23 = stablehlo.multiply %22, %13 : tensor<512x56x56x384xf32>
    %cst_6 = stablehlo.constant dense<0.112835854> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %25 = stablehlo.add %23, %24 : tensor<512x56x56x384xf32>
    %26 = stablehlo.multiply %25, %13 : tensor<512x56x56x384xf32>
    %cst_7 = stablehlo.constant dense<-0.37612626> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %28 = stablehlo.add %26, %27 : tensor<512x56x56x384xf32>
    %29 = stablehlo.multiply %28, %13 : tensor<512x56x56x384xf32>
    %cst_8 = stablehlo.constant dense<1.12837911> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %31 = stablehlo.add %29, %30 : tensor<512x56x56x384xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<512x56x56x384xf32>
    %33 = stablehlo.subtract %10, %32 : tensor<512x56x56x384xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %35 = stablehlo.compare LT, %8, %34 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %37 = stablehlo.negate %13 : tensor<512x56x56x384xf32>
    %cst_11 = stablehlo.constant dense<-88.7228394> : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %39 = stablehlo.compare LT, %37, %38 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %40 = stablehlo.exponential %37 : tensor<512x56x56x384xf32>
    %41 = stablehlo.divide %10, %9 : tensor<512x56x56x384xf32>
    %42 = stablehlo.multiply %40, %41 : tensor<512x56x56x384xf32>
    %43 = stablehlo.compare LT, %9, %36 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
    %cst_12 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %45 = stablehlo.divide %10, %13 : tensor<512x56x56x384xf32>
    %46 = stablehlo.multiply %44, %45 : tensor<512x56x56x384xf32>
    %cst_13 = stablehlo.constant dense<-0.138703942> : tensor<f32>
    %47 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %48 = stablehlo.add %46, %47 : tensor<512x56x56x384xf32>
    %49 = stablehlo.multiply %48, %45 : tensor<512x56x56x384xf32>
    %cst_14 = stablehlo.constant dense<0.368742466> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %51 = stablehlo.add %49, %50 : tensor<512x56x56x384xf32>
    %52 = stablehlo.multiply %51, %45 : tensor<512x56x56x384xf32>
    %cst_15 = stablehlo.constant dense<-0.582473278> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %54 = stablehlo.add %52, %53 : tensor<512x56x56x384xf32>
    %55 = stablehlo.multiply %54, %45 : tensor<512x56x56x384xf32>
    %cst_16 = stablehlo.constant dense<0.621000468> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %57 = stablehlo.add %55, %56 : tensor<512x56x56x384xf32>
    %58 = stablehlo.multiply %57, %45 : tensor<512x56x56x384xf32>
    %cst_17 = stablehlo.constant dense<-0.494451523> : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %60 = stablehlo.add %58, %59 : tensor<512x56x56x384xf32>
    %61 = stablehlo.multiply %60, %45 : tensor<512x56x56x384xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %63 = stablehlo.add %61, %62 : tensor<512x56x56x384xf32>
    %64 = stablehlo.multiply %63, %45 : tensor<512x56x56x384xf32>
    %cst_19 = stablehlo.constant dense<-0.274112701> : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %66 = stablehlo.add %64, %65 : tensor<512x56x56x384xf32>
    %67 = stablehlo.multiply %66, %45 : tensor<512x56x56x384xf32>
    %cst_20 = stablehlo.constant dense<0.563825965> : tensor<f32>
    %68 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %69 = stablehlo.add %67, %68 : tensor<512x56x56x384xf32>
    %cst_21 = stablehlo.constant dense<-10.477664> : tensor<f32>
    %70 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %71 = stablehlo.multiply %70, %45 : tensor<512x56x56x384xf32>
    %cst_22 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %73 = stablehlo.add %71, %72 : tensor<512x56x56x384xf32>
    %74 = stablehlo.multiply %73, %45 : tensor<512x56x56x384xf32>
    %cst_23 = stablehlo.constant dense<-7.49551868> : tensor<f32>
    %75 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %76 = stablehlo.add %74, %75 : tensor<512x56x56x384xf32>
    %77 = stablehlo.multiply %76, %45 : tensor<512x56x56x384xf32>
    %cst_24 = stablehlo.constant dense<2.92101908> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %79 = stablehlo.add %77, %78 : tensor<512x56x56x384xf32>
    %80 = stablehlo.multiply %79, %45 : tensor<512x56x56x384xf32>
    %cst_25 = stablehlo.constant dense<-1.01526523> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %82 = stablehlo.add %80, %81 : tensor<512x56x56x384xf32>
    %83 = stablehlo.multiply %82, %45 : tensor<512x56x56x384xf32>
    %cst_26 = stablehlo.constant dense<0.42184633> : tensor<f32>
    %84 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %85 = stablehlo.add %83, %84 : tensor<512x56x56x384xf32>
    %86 = stablehlo.multiply %85, %45 : tensor<512x56x56x384xf32>
    %cst_27 = stablehlo.constant dense<-0.282076746> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %88 = stablehlo.add %86, %87 : tensor<512x56x56x384xf32>
    %89 = stablehlo.multiply %88, %45 : tensor<512x56x56x384xf32>
    %cst_28 = stablehlo.constant dense<0.564189494> : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
    %91 = stablehlo.add %89, %90 : tensor<512x56x56x384xf32>
    %92 = stablehlo.select %43, %69, %91 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %93 = stablehlo.multiply %42, %92 : tensor<512x56x56x384xf32>
    %94 = stablehlo.select %39, %34, %93 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %95 = stablehlo.subtract %36, %94 : tensor<512x56x56x384xf32>
    %96 = stablehlo.select %35, %95, %94 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %97 = stablehlo.select %11, %33, %96 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
    %98 = stablehlo.multiply %4, %97 : tensor<512x56x56x384xf32>
    return %98 : tensor<512x56x56x384xf32>
  }
  func.func private @fused_computation.264(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<512x56x56xf32>, %arg3: tensor<512x56x56xf32>, %arg4: tensor<512x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<1605632x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x56x56x96xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x96xf32>) -> tensor<1605632x96xf32>
    return %12 : tensor<1605632x96xf32>
  }
  func.func private @fused_computation.265(%arg0: tensor<512x56x56xf32>) -> tensor<512x56x56xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x56x56xf32>
    return %4 : tensor<512x56x56xf32>
  }
  func.func private @fused_computation.266(%arg0: tensor<512x56x56xf32>, %arg1: tensor<512x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x56x56x96xf32>
    return %6 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.267(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x56x56x96xf32>
    return %1 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.268(%arg0: tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %1 = mhlo.copy %0 : tensor<7x7x1x96xf32>
    return %1 : tensor<7x7x1x96xf32>
  }
  func.func private @fused_computation.269(%arg0: tensor<96xf32>, %arg1: tensor<96xf32>, %arg2: tensor<512x56x56xf32>, %arg3: tensor<512x56x56xf32>, %arg4: tensor<512x56x56x96xf32>, %arg5: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg4, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<512x56x56x96xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<512x56x56x96xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %11 = stablehlo.add %9, %10 : tensor<512x56x56x96xf32>
    return %11 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.270(%arg0: tensor<512x56x56xf32>) -> tensor<512x56x56xf32> {
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<512x56x56xf32>
    %cst_0 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.add %1, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512x56x56xf32>
    return %4 : tensor<512x56x56xf32>
  }
  func.func private @fused_computation.271(%arg0: tensor<512x56x56xf32>, %arg1: tensor<512x56x56x96xf32>, %arg2: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg1, %0 : tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.010416667> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
    %3 = stablehlo.multiply %arg0, %2 : tensor<512x56x56xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<512x56x56x96xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<512x56x56x96xf32>
    return %6 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.272(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<96xf32>) -> tensor<512x56x56x96xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512x56x56x96xf32>
    return %1 : tensor<512x56x56x96xf32>
  }
  func.func private @fused_computation.273(%arg0: tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,3,96]{1,0,2,3}"} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
    %1 = mhlo.copy %0 : tensor<4x4x3x96xf32>
    return %1 : tensor<4x4x3x96xf32>
  }
  func.func private @fused_computation.274(%arg0: tensor<512x3x224x224xf32>) -> tensor<512x224x224x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,224,224,3]{2,1,3,0}"} : (tensor<512x3x224x224xf32>) -> tensor<512x224x224x3xf32>
    %1 = mhlo.copy %0 : tensor<512x224x224x3xf32>
    return %1 : tensor<512x224x224x3xf32>
  }
  func.func private @region_2.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    return %0 : tensor<512x56x56x3xf32>
  }
  func.func private @region_2.3.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.1(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    return %0 : tensor<512x56x56x3xf32>
  }
  func.func private @wrapped_multiply_computation(%arg0: tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x56x56x384xf32>
    return %0 : tensor<512x56x56x384xf32>
  }
  func.func private @region_4.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.2(%arg0: tensor<512x56x56x384xf32>, %arg1: tensor<f32>) -> tensor<512x2x2x384xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
    return %0 : tensor<512x2x2x384xf32>
  }
  func.func private @wrapped_sqrt_computation(%arg0: tensor<512x384xf32>) -> tensor<512x384xf32> {
    %0 = stablehlo.sqrt %arg0 : tensor<512x384xf32>
    return %0 : tensor<512x384xf32>
  }
  func.func private @region_5.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.3(%arg0: tensor<512x1x1x384xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
    return %0 : tensor<512x1x1x12xf32>
  }
  func.func private @region_5.8.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation(%arg0: tensor<512x1x1x12xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_2.3.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.4(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    return %0 : tensor<512x56x56x3xf32>
  }
  func.func private @wrapped_multiply_computation.1(%arg0: tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x56x56x384xf32>
    return %0 : tensor<512x56x56x384xf32>
  }
  func.func private @region_7.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.5(%arg0: tensor<512x56x56x384xf32>, %arg1: tensor<f32>) -> tensor<512x2x2x384xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
    return %0 : tensor<512x2x2x384xf32>
  }
  func.func private @wrapped_sqrt_computation.1(%arg0: tensor<512x384xf32>) -> tensor<512x384xf32> {
    %0 = stablehlo.sqrt %arg0 : tensor<512x384xf32>
    return %0 : tensor<512x384xf32>
  }
  func.func private @region_8.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.6(%arg0: tensor<512x1x1x384xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
    return %0 : tensor<512x1x1x12xf32>
  }
  func.func private @region_8.11.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.1(%arg0: tensor<512x1x1x12xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_2.3.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.7(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    return %0 : tensor<512x56x56x3xf32>
  }
  func.func private @wrapped_multiply_computation.2(%arg0: tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<512x56x56x384xf32>
    return %0 : tensor<512x56x56x384xf32>
  }
  func.func private @region_10.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.8(%arg0: tensor<512x56x56x384xf32>, %arg1: tensor<f32>) -> tensor<512x2x2x384xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
    return %0 : tensor<512x2x2x384xf32>
  }
  func.func private @wrapped_sqrt_computation.2(%arg0: tensor<512x384xf32>) -> tensor<512x384xf32> {
    %0 = stablehlo.sqrt %arg0 : tensor<512x384xf32>
    return %0 : tensor<512x384xf32>
  }
  func.func private @region_11.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.9(%arg0: tensor<512x1x1x384xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
    return %0 : tensor<512x1x1x12xf32>
  }
  func.func private @region_11.14.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.2(%arg0: tensor<512x1x1x12xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_2.3.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.10(%arg0: tensor<512x56x56x96xf32>, %arg1: tensor<f32>) -> tensor<512x56x56x3xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    return %0 : tensor<512x56x56x3xf32>
  }
  func.func private @region_15.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.11(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    return %0 : tensor<512x28x28x6xf32>
  }
  func.func private @region_17.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.12(%arg0: tensor<512x1x1x768xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
    return %0 : tensor<512x1x1x24xf32>
  }
  func.func private @region_17.22.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.3(%arg0: tensor<512x1x1x24xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_15.18.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.13(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    return %0 : tensor<512x28x28x6xf32>
  }
  func.func private @region_20.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.14(%arg0: tensor<512x1x1x768xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
    return %0 : tensor<512x1x1x24xf32>
  }
  func.func private @region_20.25.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.4(%arg0: tensor<512x1x1x24xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_15.18.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.15(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    return %0 : tensor<512x28x28x6xf32>
  }
  func.func private @region_23.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.16(%arg0: tensor<512x1x1x768xf32>, %arg1: tensor<f32>) -> tensor<512x1x1x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
    return %0 : tensor<512x1x1x24xf32>
  }
  func.func private @region_23.28.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.5(%arg0: tensor<512x1x1x24xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_15.18.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.17(%arg0: tensor<512x28x28x192xf32>, %arg1: tensor<f32>) -> tensor<512x28x28x6xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    return %0 : tensor<512x28x28x6xf32>
  }
  func.func private @region_27.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.18(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_29.36.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.6(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.19(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_32.39.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.7(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.20(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_35.42.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.8(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.21(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_38.45.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.9(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.22(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_41.48.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.10(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.23(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_44.51.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.11(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.24(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_47.54.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.12(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.25(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_50.57.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.13(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.26(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_53.60.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.14(%arg0: tensor<512x1x1x2xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_27.32.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.27(%arg0: tensor<512x14x14x384xf32>, %arg1: tensor<f32>) -> tensor<512x14x14x12xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    return %0 : tensor<512x14x14x12xf32>
  }
  func.func private @region_57.64(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.28(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x7x7x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
    return %0 : tensor<512x7x7x24xf32>
  }
  func.func private @region_59.68.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.15(%arg0: tensor<512x1x1x3xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_57.64.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.29(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x7x7x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
    return %0 : tensor<512x7x7x24xf32>
  }
  func.func private @region_62.71.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.16(%arg0: tensor<512x1x1x3xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_57.64.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.30(%arg0: tensor<512x7x7x768xf32>, %arg1: tensor<f32>) -> tensor<512x7x7x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
    return %0 : tensor<512x7x7x24xf32>
  }
  func.func private @region_65.74.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.17(%arg0: tensor<512x1x1x3xf32>, %arg1: tensor<f32>) -> tensor<512x1x1xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    return %0 : tensor<512x1x1xf32>
  }
  func.func private @region_67.76(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.31(%arg0: tensor<512x768xf32>, %arg1: tensor<f32>) -> tensor<512x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x768xf32>, tensor<f32>) -> tensor<512x24xf32>
    return %0 : tensor<512x24xf32>
  }
  func.func private @region_67.76.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_computation.18(%arg0: tensor<512x24xf32>, %arg1: tensor<f32>) -> tensor<512xf32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<512x24xf32>, tensor<f32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
  func.func private @region_69.78(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @wrapped_reduce_window_computation.32(%arg0: tensor<512x768xf32>, %arg1: tensor<f32>) -> tensor<512x24xf32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<512x768xf32>, tensor<f32>) -> tensor<512x24xf32>
    return %0 : tensor<512x24xf32>
  }
  func.func @main(%arg0: tensor<96x3x4x4xf32>, %arg1: tensor<96xf32>, %arg2: tensor<96xf32>, %arg3: tensor<96xf32>, %arg4: tensor<96x1x7x7xf32>, %arg5: tensor<96xf32>, %arg6: tensor<96xf32>, %arg7: tensor<96xf32>, %arg8: tensor<384x96xf32>, %arg9: tensor<384xf32>, %arg10: tensor<1x1x1x384xf32>, %arg11: tensor<1x1x1x384xf32>, %arg12: tensor<96x384xf32>, %arg13: tensor<96xf32>, %arg14: tensor<96x1x7x7xf32>, %arg15: tensor<96xf32>, %arg16: tensor<96xf32>, %arg17: tensor<96xf32>, %arg18: tensor<384x96xf32>, %arg19: tensor<384xf32>, %arg20: tensor<1x1x1x384xf32>, %arg21: tensor<1x1x1x384xf32>, %arg22: tensor<96x384xf32>, %arg23: tensor<96xf32>, %arg24: tensor<96x1x7x7xf32>, %arg25: tensor<96xf32>, %arg26: tensor<96xf32>, %arg27: tensor<96xf32>, %arg28: tensor<384x96xf32>, %arg29: tensor<384xf32>, %arg30: tensor<1x1x1x384xf32>, %arg31: tensor<1x1x1x384xf32>, %arg32: tensor<96x384xf32>, %arg33: tensor<96xf32>, %arg34: tensor<96xf32>, %arg35: tensor<96xf32>, %arg36: tensor<192x96x2x2xf32>, %arg37: tensor<192xf32>, %arg38: tensor<192x1x7x7xf32>, %arg39: tensor<192xf32>, %arg40: tensor<192xf32>, %arg41: tensor<192xf32>, %arg42: tensor<768x192xf32>, %arg43: tensor<768xf32>, %arg44: tensor<1x1x1x768xf32>, %arg45: tensor<1x1x1x768xf32>, %arg46: tensor<192x768xf32>, %arg47: tensor<192xf32>, %arg48: tensor<192x1x7x7xf32>, %arg49: tensor<192xf32>, %arg50: tensor<192xf32>, %arg51: tensor<192xf32>, %arg52: tensor<768x192xf32>, %arg53: tensor<768xf32>, %arg54: tensor<1x1x1x768xf32>, %arg55: tensor<1x1x1x768xf32>, %arg56: tensor<192x768xf32>, %arg57: tensor<192xf32>, %arg58: tensor<192x1x7x7xf32>, %arg59: tensor<192xf32>, %arg60: tensor<192xf32>, %arg61: tensor<192xf32>, %arg62: tensor<768x192xf32>, %arg63: tensor<768xf32>, %arg64: tensor<1x1x1x768xf32>, %arg65: tensor<1x1x1x768xf32>, %arg66: tensor<192x768xf32>, %arg67: tensor<192xf32>, %arg68: tensor<192xf32>, %arg69: tensor<192xf32>, %arg70: tensor<384x192x2x2xf32>, %arg71: tensor<384xf32>, %arg72: tensor<384x1x7x7xf32>, %arg73: tensor<384xf32>, %arg74: tensor<384xf32>, %arg75: tensor<384xf32>, %arg76: tensor<1536x384xf32>, %arg77: tensor<1536xf32>, %arg78: tensor<1x1x1x1536xf32>, %arg79: tensor<1x1x1x1536xf32>, %arg80: tensor<384x1536xf32>, %arg81: tensor<384xf32>, %arg82: tensor<384x1x7x7xf32>, %arg83: tensor<384xf32>, %arg84: tensor<384xf32>, %arg85: tensor<384xf32>, %arg86: tensor<1536x384xf32>, %arg87: tensor<1536xf32>, %arg88: tensor<1x1x1x1536xf32>, %arg89: tensor<1x1x1x1536xf32>, %arg90: tensor<384x1536xf32>, %arg91: tensor<384xf32>, %arg92: tensor<384x1x7x7xf32>, %arg93: tensor<384xf32>, %arg94: tensor<384xf32>, %arg95: tensor<384xf32>, %arg96: tensor<1536x384xf32>, %arg97: tensor<1536xf32>, %arg98: tensor<1x1x1x1536xf32>, %arg99: tensor<1x1x1x1536xf32>, %arg100: tensor<384x1536xf32>, %arg101: tensor<384xf32>, %arg102: tensor<384x1x7x7xf32>, %arg103: tensor<384xf32>, %arg104: tensor<384xf32>, %arg105: tensor<384xf32>, %arg106: tensor<1536x384xf32>, %arg107: tensor<1536xf32>, %arg108: tensor<1x1x1x1536xf32>, %arg109: tensor<1x1x1x1536xf32>, %arg110: tensor<384x1536xf32>, %arg111: tensor<384xf32>, %arg112: tensor<384x1x7x7xf32>, %arg113: tensor<384xf32>, %arg114: tensor<384xf32>, %arg115: tensor<384xf32>, %arg116: tensor<1536x384xf32>, %arg117: tensor<1536xf32>, %arg118: tensor<1x1x1x1536xf32>, %arg119: tensor<1x1x1x1536xf32>, %arg120: tensor<384x1536xf32>, %arg121: tensor<384xf32>, %arg122: tensor<384x1x7x7xf32>, %arg123: tensor<384xf32>, %arg124: tensor<384xf32>, %arg125: tensor<384xf32>, %arg126: tensor<1536x384xf32>, %arg127: tensor<1536xf32>, %arg128: tensor<1x1x1x1536xf32>, %arg129: tensor<1x1x1x1536xf32>, %arg130: tensor<384x1536xf32>, %arg131: tensor<384xf32>, %arg132: tensor<384x1x7x7xf32>, %arg133: tensor<384xf32>, %arg134: tensor<384xf32>, %arg135: tensor<384xf32>, %arg136: tensor<1536x384xf32>, %arg137: tensor<1536xf32>, %arg138: tensor<1x1x1x1536xf32>, %arg139: tensor<1x1x1x1536xf32>, %arg140: tensor<384x1536xf32>, %arg141: tensor<384xf32>, %arg142: tensor<384x1x7x7xf32>, %arg143: tensor<384xf32>, %arg144: tensor<384xf32>, %arg145: tensor<384xf32>, %arg146: tensor<1536x384xf32>, %arg147: tensor<1536xf32>, %arg148: tensor<1x1x1x1536xf32>, %arg149: tensor<1x1x1x1536xf32>, %arg150: tensor<384x1536xf32>, %arg151: tensor<384xf32>, %arg152: tensor<384x1x7x7xf32>, %arg153: tensor<384xf32>, %arg154: tensor<384xf32>, %arg155: tensor<384xf32>, %arg156: tensor<1536x384xf32>, %arg157: tensor<1536xf32>, %arg158: tensor<1x1x1x1536xf32>, %arg159: tensor<1x1x1x1536xf32>, %arg160: tensor<384x1536xf32>, %arg161: tensor<384xf32>, %arg162: tensor<384xf32>, %arg163: tensor<384xf32>, %arg164: tensor<768x384x2x2xf32>, %arg165: tensor<768xf32>, %arg166: tensor<768x1x7x7xf32>, %arg167: tensor<768xf32>, %arg168: tensor<768xf32>, %arg169: tensor<768xf32>, %arg170: tensor<3072x768xf32>, %arg171: tensor<3072xf32>, %arg172: tensor<1x1x1x3072xf32>, %arg173: tensor<1x1x1x3072xf32>, %arg174: tensor<768x3072xf32>, %arg175: tensor<768xf32>, %arg176: tensor<768x1x7x7xf32>, %arg177: tensor<768xf32>, %arg178: tensor<768xf32>, %arg179: tensor<768xf32>, %arg180: tensor<3072x768xf32>, %arg181: tensor<3072xf32>, %arg182: tensor<1x1x1x3072xf32>, %arg183: tensor<1x1x1x3072xf32>, %arg184: tensor<768x3072xf32>, %arg185: tensor<768xf32>, %arg186: tensor<768x1x7x7xf32>, %arg187: tensor<768xf32>, %arg188: tensor<768xf32>, %arg189: tensor<768xf32>, %arg190: tensor<3072x768xf32>, %arg191: tensor<3072xf32>, %arg192: tensor<1x1x1x3072xf32>, %arg193: tensor<1x1x1x3072xf32>, %arg194: tensor<768x3072xf32>, %arg195: tensor<768xf32>, %arg196: tensor<768xf32>, %arg197: tensor<768xf32>, %arg198: tensor<1000x768xf32>, %arg199: tensor<1000xf32>, %arg200: tensor<512x3x224x224xf32>) -> tensor<512x1000xf32> {
    %0 = "mhlo.fusion"(%arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x3x224x224xf32>):
      %398 = stablehlo.transpose %arg201, dims = [0, 2, 3, 1] {result_layout = dense<[2, 1, 3, 0]> : tensor<4xindex>, xla_shape = "f32[512,224,224,3]{2,1,3,0}"} : (tensor<512x3x224x224xf32>) -> tensor<512x224x224x3xf32>
      %399 = mhlo.copy %398 : tensor<512x224x224x3xf32>
      mhlo.return %399 : tensor<512x224x224x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x3x224x224xf32>) -> tensor<512x224x224x3xf32>
    %1 = "mhlo.fusion"(%arg0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x3x4x4xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[4,4,3,96]{1,0,2,3}"} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
      %399 = mhlo.copy %398 : tensor<4x4x3x96xf32>
      mhlo.return %399 : tensor<4x4x3x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x3x4x4xf32>) -> tensor<4x4x3x96xf32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x224x224x3xf32>, tensor<4x4x3x96xf32>) -> tensor<512x56x56x96xf32>
    %3 = "mhlo.fusion"(%2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x56x56x96xf32>
      mhlo.return %399 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.fusion"(%3, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %5 = "mhlo.fusion"(%4, %2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>, %arg202: tensor<512x56x56x96xf32>, %arg203: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x56x56x96xf32>
      mhlo.return %404 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %6 = "mhlo.fusion"(%5, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
      mhlo.return %398 : tensor<512x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    %7 = "mhlo.fusion"(%6, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %8 = "mhlo.fusion"(%7) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x56x56xf32>
      mhlo.return %402 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>) -> tensor<512x56x56xf32>
    %9 = "mhlo.fusion"(%arg3, %arg2, %8, %4, %2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<512x56x56xf32>, %arg204: tensor<512x56x56xf32>, %arg205: tensor<512x56x56x96xf32>, %arg206: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x56x56x96xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x56x56x96xf32>
      mhlo.return %409 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<512x56x56xf32>, tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %10 = "mhlo.fusion"(%arg4) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x96xf32>
      mhlo.return %399 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %11 = stablehlo.convolution(%9, %10) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<512x56x56x96xf32>
    %12 = "mhlo.fusion"(%11, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x56x56x96xf32>
      mhlo.return %399 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %13 = "mhlo.fusion"(%12, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %14 = "mhlo.fusion"(%13, %11, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>, %arg202: tensor<512x56x56x96xf32>, %arg203: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x56x56x96xf32>
      mhlo.return %404 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %15 = "mhlo.fusion"(%14, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
      mhlo.return %398 : tensor<512x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    %16 = "mhlo.fusion"(%15, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %17 = "mhlo.fusion"(%16) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x56x56xf32>
      mhlo.return %402 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>) -> tensor<512x56x56xf32>
    %18 = "mhlo.fusion"(%arg7, %arg6, %17, %13, %11, %arg5) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<512x56x56xf32>, %arg204: tensor<512x56x56xf32>, %arg205: tensor<512x56x56x96xf32>, %arg206: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x56x56x96xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x56x56x96xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x96xf32>) -> tensor<1605632x96xf32>
      mhlo.return %410 : tensor<1605632x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<512x56x56xf32>, tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<1605632x96xf32>
    %19 = stablehlo.dot_general %18, %arg8, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1605632x96xf32>, tensor<384x96xf32>) -> tensor<1605632x384xf32>
    %20 = "mhlo.fusion"(%19, %arg9) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1605632x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1605632x384xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<1605632x384xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<1605632x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
      %403 = stablehlo.negate %400 : tensor<1605632x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<1605632x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
      %407 = stablehlo.abs %406 : tensor<512x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x56x56x384xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x56x56x384xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x56x56x384xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x56x56x384xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x56x56x384xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x56x56x384xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x56x56x384xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x56x56x384xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %435 = stablehlo.negate %411 : tensor<512x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %438 = stablehlo.exponential %435 : tensor<512x56x56x384xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x56x56x384xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x56x56x384xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x56x56x384xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x56x56x384xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x56x56x384xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x56x56x384xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x56x56x384xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x56x56x384xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x56x56x384xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x56x56x384xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x56x56x384xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x56x56x384xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x56x56x384xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x56x56x384xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x56x56x384xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x56x56x384xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x56x56x384xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x56x56x384xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x56x56x384xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x56x56x384xf32>
      mhlo.return %496 : tensor<512x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1605632x384xf32>, tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %21 = "mhlo.fusion"(%20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x56x56x384xf32>
      mhlo.return %398 : tensor<512x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xf32>
    %22 = "mhlo.fusion"(%21, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
      mhlo.return %398 : tensor<512x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
    %23 = "mhlo.fusion"(%22, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x2x2x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
      mhlo.return %398 : tensor<512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
    %24 = "mhlo.fusion"(%23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x384xf32>):
      %398 = stablehlo.sqrt %arg201 : tensor<512x384xf32>
      mhlo.return %398 : tensor<512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x384xf32>) -> tensor<512x384xf32>
    %25 = mhlo.bitcast %24 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %26 = "mhlo.fusion"(%25, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
      mhlo.return %398 : tensor<512x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
    %27 = "mhlo.fusion"(%26, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %28 = "mhlo.fusion"(%27, %24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x384xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %400 = stablehlo.multiply %arg201, %399 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %402 = stablehlo.add %400, %401 : tensor<512x1x1xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x384xf32>
      %405 = stablehlo.divide %398, %404 : tensor<512x1x1x384xf32>
      mhlo.return %405 : tensor<512x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %29 = "mhlo.fusion"(%20, %arg11, %28, %arg10) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<512x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x384xf32>) -> tensor<512x384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x384xf32>) -> tensor<512x56x56x384xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x56x56x384xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x56x56x384xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x56x56x384xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x56x56x384xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x384xf32>) -> tensor<1605632x384xf32>
      mhlo.return %408 : tensor<1605632x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<512x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1605632x384xf32>
    %30 = stablehlo.dot_general %29, %arg12, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1605632x384xf32>, tensor<96x384xf32>) -> tensor<1605632x96xf32>
    %31 = "mhlo.fusion"(%30, %arg13, %arg3, %arg2, %8, %4, %2, %arg1) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1605632x96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<96xf32>, %arg204: tensor<96xf32>, %arg205: tensor<512x56x56xf32>, %arg206: tensor<512x56x56xf32>, %arg207: tensor<512x56x56x96xf32>, %arg208: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg208, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg207, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg206, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg205, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x56x56x96xf32>
      %408 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x56x56x96xf32>
      %410 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<96xf32>) -> tensor<1605632x96xf32>
      %411 = stablehlo.add %410, %arg201 : tensor<1605632x96xf32>
      %412 = mhlo.bitcast %411 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x96xf32>) -> tensor<512x56x56x96xf32>
      %413 = stablehlo.add %409, %412 : tensor<512x56x56x96xf32>
      mhlo.return %413 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<1605632x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<512x56x56xf32>, tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %32 = "mhlo.fusion"(%arg14) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x96xf32>
      mhlo.return %399 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %33 = stablehlo.convolution(%31, %32) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<512x56x56x96xf32>
    %34 = "mhlo.fusion"(%33, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x56x56x96xf32>
      mhlo.return %399 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %35 = "mhlo.fusion"(%34, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %36 = "mhlo.fusion"(%35, %33, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>, %arg202: tensor<512x56x56x96xf32>, %arg203: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x56x56x96xf32>
      mhlo.return %404 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %37 = "mhlo.fusion"(%36, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
      mhlo.return %398 : tensor<512x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    %38 = "mhlo.fusion"(%37, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %39 = "mhlo.fusion"(%38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x56x56xf32>
      mhlo.return %402 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>) -> tensor<512x56x56xf32>
    %40 = "mhlo.fusion"(%arg17, %arg16, %39, %35, %33, %arg15) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<512x56x56xf32>, %arg204: tensor<512x56x56xf32>, %arg205: tensor<512x56x56x96xf32>, %arg206: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x56x56x96xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x56x56x96xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x96xf32>) -> tensor<1605632x96xf32>
      mhlo.return %410 : tensor<1605632x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<512x56x56xf32>, tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<1605632x96xf32>
    %41 = stablehlo.dot_general %40, %arg18, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1605632x96xf32>, tensor<384x96xf32>) -> tensor<1605632x384xf32>
    %42 = "mhlo.fusion"(%41, %arg19) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1605632x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1605632x384xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<1605632x384xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<1605632x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
      %403 = stablehlo.negate %400 : tensor<1605632x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<1605632x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
      %407 = stablehlo.abs %406 : tensor<512x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x56x56x384xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x56x56x384xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x56x56x384xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x56x56x384xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x56x56x384xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x56x56x384xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x56x56x384xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x56x56x384xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %435 = stablehlo.negate %411 : tensor<512x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %438 = stablehlo.exponential %435 : tensor<512x56x56x384xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x56x56x384xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x56x56x384xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x56x56x384xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x56x56x384xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x56x56x384xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x56x56x384xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x56x56x384xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x56x56x384xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x56x56x384xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x56x56x384xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x56x56x384xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x56x56x384xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x56x56x384xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x56x56x384xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x56x56x384xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x56x56x384xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x56x56x384xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x56x56x384xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x56x56x384xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x56x56x384xf32>
      mhlo.return %496 : tensor<512x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1605632x384xf32>, tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %43 = "mhlo.fusion"(%42) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x56x56x384xf32>
      mhlo.return %398 : tensor<512x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xf32>
    %44 = "mhlo.fusion"(%43, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
      mhlo.return %398 : tensor<512x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
    %45 = "mhlo.fusion"(%44, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x2x2x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
      mhlo.return %398 : tensor<512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
    %46 = "mhlo.fusion"(%45) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x384xf32>):
      %398 = stablehlo.sqrt %arg201 : tensor<512x384xf32>
      mhlo.return %398 : tensor<512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x384xf32>) -> tensor<512x384xf32>
    %47 = mhlo.bitcast %46 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %48 = "mhlo.fusion"(%47, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
      mhlo.return %398 : tensor<512x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
    %49 = "mhlo.fusion"(%48, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %50 = "mhlo.fusion"(%49, %46) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x384xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %400 = stablehlo.multiply %arg201, %399 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %402 = stablehlo.add %400, %401 : tensor<512x1x1xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x384xf32>
      %405 = stablehlo.divide %398, %404 : tensor<512x1x1x384xf32>
      mhlo.return %405 : tensor<512x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %51 = "mhlo.fusion"(%42, %arg21, %50, %arg20) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<512x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x384xf32>) -> tensor<512x384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x384xf32>) -> tensor<512x56x56x384xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x56x56x384xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x56x56x384xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x56x56x384xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x56x56x384xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x384xf32>) -> tensor<1605632x384xf32>
      mhlo.return %408 : tensor<1605632x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<512x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1605632x384xf32>
    %52 = stablehlo.dot_general %51, %arg22, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1605632x384xf32>, tensor<96x384xf32>) -> tensor<1605632x96xf32>
    %53 = "mhlo.fusion"(%31, %52, %arg23) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<1605632x96xf32>, %arg203: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<96xf32>) -> tensor<1605632x96xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<1605632x96xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x96xf32>) -> tensor<512x56x56x96xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x56x56x96xf32>
      mhlo.return %401 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<1605632x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %54 = "mhlo.fusion"(%arg24) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,96]{1,0,2,3}"} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x96xf32>
      mhlo.return %399 : tensor<7x7x1x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96x1x7x7xf32>) -> tensor<7x7x1x96xf32>
    %55 = stablehlo.convolution(%53, %54) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 96 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x56x56x96xf32>, tensor<7x7x1x96xf32>) -> tensor<512x56x56x96xf32>
    %56 = "mhlo.fusion"(%55, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x56x56x96xf32>
      mhlo.return %399 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %57 = "mhlo.fusion"(%56, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %58 = "mhlo.fusion"(%57, %55, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>, %arg202: tensor<512x56x56x96xf32>, %arg203: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x56x56x96xf32>
      mhlo.return %404 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %59 = "mhlo.fusion"(%58, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
      mhlo.return %398 : tensor<512x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    %60 = "mhlo.fusion"(%59, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x56x56xf32>
      mhlo.return %402 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>) -> tensor<512x56x56xf32>
    %62 = "mhlo.fusion"(%arg27, %arg26, %61, %57, %55, %arg25) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<512x56x56xf32>, %arg204: tensor<512x56x56xf32>, %arg205: tensor<512x56x56x96xf32>, %arg206: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x56x56x96xf32>
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x56x56x96xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x56x56x96xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x96xf32>) -> tensor<1605632x96xf32>
      mhlo.return %410 : tensor<1605632x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<512x56x56xf32>, tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<96xf32>) -> tensor<1605632x96xf32>
    %63 = stablehlo.dot_general %62, %arg28, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1605632x96xf32>, tensor<384x96xf32>) -> tensor<1605632x384xf32>
    %64 = "mhlo.fusion"(%63, %arg29) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<1605632x384xf32>, %arg202: tensor<384xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<1605632x384xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<1605632x384xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<1605632x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
      %403 = stablehlo.negate %400 : tensor<1605632x384xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1605632x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<1605632x384xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x384xf32>) -> tensor<512x56x56x384xf32>
      %407 = stablehlo.abs %406 : tensor<512x56x56x384xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x56x56x384xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x56x56x384xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x56x56x384xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x56x56x384xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x56x56x384xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x56x56x384xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x56x56x384xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x56x56x384xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x56x56x384xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x56x56x384xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x56x56x384xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x56x56x384xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x56x56x384xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x56x56x384xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x56x56x384xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %435 = stablehlo.negate %411 : tensor<512x56x56x384xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %438 = stablehlo.exponential %435 : tensor<512x56x56x384xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x56x56x384xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x56x56x384xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x56x56x384xf32>, tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x56x56x384xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x56x56x384xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x56x56x384xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x56x56x384xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x56x56x384xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x56x56x384xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x56x56x384xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x56x56x384xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x56x56x384xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x56x56x384xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x56x56x384xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x56x56x384xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x56x56x384xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x56x56x384xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x56x56x384xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x56x56x384xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x56x56x384xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x56x56x384xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x56x56x384xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x56x56x384xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x56x56x384xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x56x56x384xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x56x56x384xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x56x56x384xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x56x56x384xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x56x56x384xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x56x56x384xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x56x56x384xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x56x56x384xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x56x56x384xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x56x56x384xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x56x56x384xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x56x56x384xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x56x56x384xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x56x56x384xi1>, tensor<512x56x56x384xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x56x56x384xf32>
      mhlo.return %496 : tensor<512x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<1605632x384xf32>, tensor<384xf32>) -> tensor<512x56x56x384xf32>
    %65 = "mhlo.fusion"(%64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x56x56x384xf32>
      mhlo.return %398 : tensor<512x56x56x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>) -> tensor<512x56x56x384xf32>
    %66 = "mhlo.fusion"(%65, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [4, 4], [4, 4], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 32, 32, 1>, window_strides = array<i64: 1, 32, 32, 1>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
      mhlo.return %398 : tensor<512x2x2x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>, tensor<f32>) -> tensor<512x2x2x384xf32>
    %67 = "mhlo.fusion"(%66, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x2x2x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
      mhlo.return %398 : tensor<512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x2x2x384xf32>, tensor<f32>) -> tensor<512x384xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x384xf32>):
      %398 = stablehlo.sqrt %arg201 : tensor<512x384xf32>
      mhlo.return %398 : tensor<512x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x384xf32>) -> tensor<512x384xf32>
    %69 = mhlo.bitcast %68 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %70 = "mhlo.fusion"(%69, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
      mhlo.return %398 : tensor<512x1x1x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x384xf32>, tensor<f32>) -> tensor<512x1x1x12xf32>
    %71 = "mhlo.fusion"(%70, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x12xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %72 = "mhlo.fusion"(%71, %68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x384xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %400 = stablehlo.multiply %arg201, %399 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %402 = stablehlo.add %400, %401 : tensor<512x1x1xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x384xf32>
      %405 = stablehlo.divide %398, %404 : tensor<512x1x1x384xf32>
      mhlo.return %405 : tensor<512x1x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x384xf32>) -> tensor<512x1x1x384xf32>
    %73 = "mhlo.fusion"(%64, %arg31, %72, %arg30) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x384xf32>, %arg202: tensor<1x1x1x384xf32>, %arg203: tensor<512x1x1x384xf32>, %arg204: tensor<1x1x1x384xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x384xf32>) -> tensor<512x384xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x384xf32>) -> tensor<512x56x56x384xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x56x56x384xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x56x56x384xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x384xf32>) -> tensor<384xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<384xf32>) -> tensor<512x56x56x384xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x56x56x384xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x56x56x384xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x56x56x384xf32>) -> tensor<1605632x384xf32>
      mhlo.return %408 : tensor<1605632x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x384xf32>, tensor<1x1x1x384xf32>, tensor<512x1x1x384xf32>, tensor<1x1x1x384xf32>) -> tensor<1605632x384xf32>
    %74 = stablehlo.dot_general %73, %arg32, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1605632x384xf32>, tensor<96x384xf32>) -> tensor<1605632x96xf32>
    %75 = "mhlo.fusion"(%53, %74, %arg33) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<1605632x96xf32>, %arg203: tensor<96xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<96xf32>) -> tensor<1605632x96xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<1605632x96xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<1605632x96xf32>) -> tensor<512x56x56x96xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x56x56x96xf32>
      mhlo.return %401 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<1605632x96xf32>, tensor<96xf32>) -> tensor<512x56x56x96xf32>
    %76 = "mhlo.fusion"(%75, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %77 = "mhlo.fusion"(%75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg202, %398 : tensor<512x56x56xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %401 = stablehlo.subtract %arg201, %400 : tensor<512x56x56x96xf32>
      %402 = stablehlo.multiply %401, %401 : tensor<512x56x56x96xf32>
      mhlo.return %402 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %78 = "mhlo.fusion"(%77, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56x96xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
      mhlo.return %398 : tensor<512x56x56x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x96xf32>, tensor<f32>) -> tensor<512x56x56x3xf32>
    %79 = "mhlo.fusion"(%78, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x56x56x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
      mhlo.return %398 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56x3xf32>, tensor<f32>) -> tensor<512x56x56xf32>
    %80 = "mhlo.fusion"(%79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x56x56xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x56x56xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x56x56xf32>
      mhlo.return %402 : tensor<512x56x56xf32>
    }) {output_operand_aliasing = []} : (tensor<512x56x56xf32>) -> tensor<512x56x56xf32>
    %81 = "mhlo.fusion"(%arg35, %arg34, %80, %75, %76) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<96xf32>, %arg202: tensor<96xf32>, %arg203: tensor<512x56x56xf32>, %arg204: tensor<512x56x56x96xf32>, %arg205: tensor<512x56x56xf32>):
      %cst_0 = stablehlo.constant dense<0.010416667> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x56x56xf32>
      %399 = stablehlo.multiply %arg205, %398 : tensor<512x56x56xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %401 = stablehlo.subtract %arg204, %400 : tensor<512x56x56x96xf32>
      %402 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<512x56x56x96xf32>
      %404 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x56x56x96xf32>
      %406 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<96xf32>) -> tensor<512x56x56x96xf32>
      %407 = stablehlo.add %405, %406 : tensor<512x56x56x96xf32>
      mhlo.return %407 : tensor<512x56x56x96xf32>
    }) {output_operand_aliasing = []} : (tensor<96xf32>, tensor<96xf32>, tensor<512x56x56xf32>, tensor<512x56x56x96xf32>, tensor<512x56x56xf32>) -> tensor<512x56x56x96xf32>
    %82 = "mhlo.fusion"(%arg36) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x96x2x2xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,96,192]{1,0,2,3}"} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
      %399 = mhlo.copy %398 : tensor<2x2x96x192xf32>
      mhlo.return %399 : tensor<2x2x96x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x96x2x2xf32>) -> tensor<2x2x96x192xf32>
    %83 = stablehlo.convolution(%81, %82) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x56x56x96xf32>, tensor<2x2x96x192xf32>) -> tensor<512x28x28x192xf32>
    %84 = "mhlo.fusion"(%83, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x28x28x192xf32>
      mhlo.return %399 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %85 = "mhlo.fusion"(%arg38) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x192xf32>
      mhlo.return %399 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %86 = stablehlo.convolution(%84, %85) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<512x28x28x192xf32>
    %87 = "mhlo.fusion"(%86, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x28x28x192xf32>
      mhlo.return %399 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %88 = "mhlo.fusion"(%87, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %89 = "mhlo.fusion"(%88, %86, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>, %arg202: tensor<512x28x28x192xf32>, %arg203: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x28x28x192xf32>
      mhlo.return %404 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %90 = "mhlo.fusion"(%89, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
      mhlo.return %398 : tensor<512x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    %91 = "mhlo.fusion"(%90, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x6xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %92 = "mhlo.fusion"(%91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x28x28xf32>
      mhlo.return %402 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>) -> tensor<512x28x28xf32>
    %93 = "mhlo.fusion"(%arg41, %arg40, %92, %88, %86, %arg39) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<512x28x28xf32>, %arg204: tensor<512x28x28xf32>, %arg205: tensor<512x28x28x192xf32>, %arg206: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x28x28x192xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x28x28x192xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x28x28x192xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x192xf32>) -> tensor<401408x192xf32>
      mhlo.return %410 : tensor<401408x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<512x28x28xf32>, tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<401408x192xf32>
    %94 = stablehlo.dot_general %93, %arg42, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<401408x192xf32>, tensor<768x192xf32>) -> tensor<401408x768xf32>
    %95 = "mhlo.fusion"(%94, %arg43) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<401408x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<401408x768xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<401408x768xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<401408x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
      %403 = stablehlo.negate %400 : tensor<401408x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<401408x768xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
      %407 = stablehlo.abs %406 : tensor<512x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x28x28x768xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x28x28x768xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x28x28x768xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x28x28x768xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x28x28x768xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x28x28x768xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x28x28x768xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x28x28x768xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %435 = stablehlo.negate %411 : tensor<512x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %438 = stablehlo.exponential %435 : tensor<512x28x28x768xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x28x28x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x28x28x768xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x28x28x768xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x28x28x768xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x28x28x768xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x28x28x768xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x28x28x768xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x28x28x768xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x28x28x768xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x28x28x768xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x28x28x768xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x28x28x768xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x28x28x768xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x28x28x768xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x28x28x768xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x28x28x768xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x28x28x768xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x28x28x768xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x28x28x768xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x28x28x768xf32>
      mhlo.return %496 : tensor<512x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<401408x768xf32>, tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %96 = "mhlo.fusion"(%95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x768xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x28x28x768xf32>, tensor<f32>) -> tensor<512x768xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x768xf32>
      mhlo.return %400 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x768xf32>) -> tensor<512x768xf32>
    %97 = mhlo.bitcast %96 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %98 = "mhlo.fusion"(%97, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
      mhlo.return %398 : tensor<512x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
    %99 = "mhlo.fusion"(%98, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %100 = "mhlo.fusion"(%99, %96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x768xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %400 = stablehlo.multiply %arg201, %399 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %402 = stablehlo.add %400, %401 : tensor<512x1x1xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x768xf32>
      %405 = stablehlo.divide %398, %404 : tensor<512x1x1x768xf32>
      mhlo.return %405 : tensor<512x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %101 = "mhlo.fusion"(%95, %arg45, %100, %arg44) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<512x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x768xf32>) -> tensor<512x768xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x768xf32>) -> tensor<512x28x28x768xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x28x28x768xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x28x28x768xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x28x28x768xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x28x28x768xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x768xf32>) -> tensor<401408x768xf32>
      mhlo.return %408 : tensor<401408x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<512x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<401408x768xf32>
    %102 = stablehlo.dot_general %101, %arg46, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<401408x768xf32>, tensor<192x768xf32>) -> tensor<401408x192xf32>
    %103 = "mhlo.fusion"(%102, %arg47, %83, %arg37) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<401408x192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<512x28x28x192xf32>, %arg204: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg203, %398 : tensor<512x28x28x192xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<192xf32>) -> tensor<401408x192xf32>
      %401 = stablehlo.add %400, %arg201 : tensor<401408x192xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x192xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.add %399, %402 : tensor<512x28x28x192xf32>
      mhlo.return %403 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<401408x192xf32>, tensor<192xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %104 = "mhlo.fusion"(%arg48) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x192xf32>
      mhlo.return %399 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %105 = stablehlo.convolution(%103, %104) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<512x28x28x192xf32>
    %106 = "mhlo.fusion"(%105, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x28x28x192xf32>
      mhlo.return %399 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %107 = "mhlo.fusion"(%106, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %108 = "mhlo.fusion"(%107, %105, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>, %arg202: tensor<512x28x28x192xf32>, %arg203: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x28x28x192xf32>
      mhlo.return %404 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %109 = "mhlo.fusion"(%108, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
      mhlo.return %398 : tensor<512x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    %110 = "mhlo.fusion"(%109, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x6xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x28x28xf32>
      mhlo.return %402 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>) -> tensor<512x28x28xf32>
    %112 = "mhlo.fusion"(%arg51, %arg50, %111, %107, %105, %arg49) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<512x28x28xf32>, %arg204: tensor<512x28x28xf32>, %arg205: tensor<512x28x28x192xf32>, %arg206: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x28x28x192xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x28x28x192xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x28x28x192xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x192xf32>) -> tensor<401408x192xf32>
      mhlo.return %410 : tensor<401408x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<512x28x28xf32>, tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<401408x192xf32>
    %113 = stablehlo.dot_general %112, %arg52, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<401408x192xf32>, tensor<768x192xf32>) -> tensor<401408x768xf32>
    %114 = "mhlo.fusion"(%113, %arg53) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<401408x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<401408x768xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<401408x768xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<401408x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
      %403 = stablehlo.negate %400 : tensor<401408x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<401408x768xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
      %407 = stablehlo.abs %406 : tensor<512x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x28x28x768xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x28x28x768xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x28x28x768xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x28x28x768xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x28x28x768xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x28x28x768xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x28x28x768xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x28x28x768xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %435 = stablehlo.negate %411 : tensor<512x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %438 = stablehlo.exponential %435 : tensor<512x28x28x768xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x28x28x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x28x28x768xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x28x28x768xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x28x28x768xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x28x28x768xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x28x28x768xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x28x28x768xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x28x28x768xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x28x28x768xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x28x28x768xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x28x28x768xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x28x28x768xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x28x28x768xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x28x28x768xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x28x28x768xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x28x28x768xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x28x28x768xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x28x28x768xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x28x28x768xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x28x28x768xf32>
      mhlo.return %496 : tensor<512x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<401408x768xf32>, tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x768xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x28x28x768xf32>, tensor<f32>) -> tensor<512x768xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x768xf32>
      mhlo.return %400 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x768xf32>) -> tensor<512x768xf32>
    %116 = mhlo.bitcast %115 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %117 = "mhlo.fusion"(%116, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
      mhlo.return %398 : tensor<512x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
    %118 = "mhlo.fusion"(%117, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %119 = "mhlo.fusion"(%118, %115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x768xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %400 = stablehlo.multiply %arg201, %399 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %402 = stablehlo.add %400, %401 : tensor<512x1x1xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x768xf32>
      %405 = stablehlo.divide %398, %404 : tensor<512x1x1x768xf32>
      mhlo.return %405 : tensor<512x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %120 = "mhlo.fusion"(%114, %arg55, %119, %arg54) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<512x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x768xf32>) -> tensor<512x768xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x768xf32>) -> tensor<512x28x28x768xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x28x28x768xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x28x28x768xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x28x28x768xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x28x28x768xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x768xf32>) -> tensor<401408x768xf32>
      mhlo.return %408 : tensor<401408x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<512x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<401408x768xf32>
    %121 = stablehlo.dot_general %120, %arg56, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<401408x768xf32>, tensor<192x768xf32>) -> tensor<401408x192xf32>
    %122 = "mhlo.fusion"(%103, %121, %arg57) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<401408x192xf32>, %arg203: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<192xf32>) -> tensor<401408x192xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<401408x192xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x192xf32>) -> tensor<512x28x28x192xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x28x28x192xf32>
      mhlo.return %401 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<401408x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %123 = "mhlo.fusion"(%arg58) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,192]{1,0,2,3}"} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x192xf32>
      mhlo.return %399 : tensor<7x7x1x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192x1x7x7xf32>) -> tensor<7x7x1x192xf32>
    %124 = stablehlo.convolution(%122, %123) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x28x28x192xf32>, tensor<7x7x1x192xf32>) -> tensor<512x28x28x192xf32>
    %125 = "mhlo.fusion"(%124, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x28x28x192xf32>
      mhlo.return %399 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %126 = "mhlo.fusion"(%125, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %127 = "mhlo.fusion"(%126, %124, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>, %arg202: tensor<512x28x28x192xf32>, %arg203: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x28x28x192xf32>
      mhlo.return %404 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %128 = "mhlo.fusion"(%127, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
      mhlo.return %398 : tensor<512x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    %129 = "mhlo.fusion"(%128, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x6xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %130 = "mhlo.fusion"(%129) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x28x28xf32>
      mhlo.return %402 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>) -> tensor<512x28x28xf32>
    %131 = "mhlo.fusion"(%arg61, %arg60, %130, %126, %124, %arg59) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<512x28x28xf32>, %arg204: tensor<512x28x28xf32>, %arg205: tensor<512x28x28x192xf32>, %arg206: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x28x28x192xf32>
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x28x28x192xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x28x28x192xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x28x28x192xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x192xf32>) -> tensor<401408x192xf32>
      mhlo.return %410 : tensor<401408x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<512x28x28xf32>, tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<192xf32>) -> tensor<401408x192xf32>
    %132 = stablehlo.dot_general %131, %arg62, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<401408x192xf32>, tensor<768x192xf32>) -> tensor<401408x768xf32>
    %133 = "mhlo.fusion"(%132, %arg63) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<401408x768xf32>, %arg202: tensor<768xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<401408x768xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<401408x768xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<401408x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
      %403 = stablehlo.negate %400 : tensor<401408x768xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<401408x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<401408x768xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x768xf32>) -> tensor<512x28x28x768xf32>
      %407 = stablehlo.abs %406 : tensor<512x28x28x768xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x28x28x768xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x28x28x768xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x28x28x768xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x28x28x768xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x28x28x768xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x28x28x768xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x28x28x768xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x28x28x768xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x28x28x768xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x28x28x768xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x28x28x768xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x28x28x768xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x28x28x768xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x28x28x768xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x28x28x768xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %435 = stablehlo.negate %411 : tensor<512x28x28x768xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %438 = stablehlo.exponential %435 : tensor<512x28x28x768xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x28x28x768xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x28x28x768xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x28x28x768xf32>, tensor<512x28x28x768xf32>) -> tensor<512x28x28x768xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x28x28x768xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x28x28x768xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x28x28x768xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x28x28x768xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x28x28x768xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x28x28x768xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x28x28x768xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x28x28x768xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x28x28x768xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x28x28x768xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x28x28x768xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x28x28x768xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x28x28x768xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x28x28x768xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x28x28x768xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x28x28x768xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x28x28x768xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x28x28x768xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x28x28x768xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x28x28x768xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x28x28x768xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x28x28x768xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x28x28x768xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x28x28x768xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x28x28x768xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x28x28x768xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x28x28x768xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x28x28x768xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x28x28x768xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x28x28x768xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x28x28x768xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x28x28x768xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x28x28x768xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x28x28x768xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x28x28x768xi1>, tensor<512x28x28x768xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x28x28x768xf32>
      mhlo.return %496 : tensor<512x28x28x768xf32>
    }) {output_operand_aliasing = []} : (tensor<401408x768xf32>, tensor<768xf32>) -> tensor<512x28x28x768xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x768xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x28x28x768xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x28x28x768xf32>, tensor<f32>) -> tensor<512x768xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x768xf32>
      mhlo.return %400 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x768xf32>) -> tensor<512x768xf32>
    %135 = mhlo.bitcast %134 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %136 = "mhlo.fusion"(%135, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
      mhlo.return %398 : tensor<512x1x1x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x768xf32>, tensor<f32>) -> tensor<512x1x1x24xf32>
    %137 = "mhlo.fusion"(%136, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x24xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %138 = "mhlo.fusion"(%137, %134) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x768xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %400 = stablehlo.multiply %arg201, %399 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %402 = stablehlo.add %400, %401 : tensor<512x1x1xf32>
      %403 = mhlo.bitcast %402 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %404 = stablehlo.broadcast_in_dim %403, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x768xf32>
      %405 = stablehlo.divide %398, %404 : tensor<512x1x1x768xf32>
      mhlo.return %405 : tensor<512x1x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x768xf32>) -> tensor<512x1x1x768xf32>
    %139 = "mhlo.fusion"(%133, %arg65, %138, %arg64) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x768xf32>, %arg202: tensor<1x1x1x768xf32>, %arg203: tensor<512x1x1x768xf32>, %arg204: tensor<1x1x1x768xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x768xf32>) -> tensor<512x768xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x768xf32>) -> tensor<512x28x28x768xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x28x28x768xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x28x28x768xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x768xf32>) -> tensor<768xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<768xf32>) -> tensor<512x28x28x768xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x28x28x768xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x28x28x768xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x28x28x768xf32>) -> tensor<401408x768xf32>
      mhlo.return %408 : tensor<401408x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x768xf32>, tensor<1x1x1x768xf32>, tensor<512x1x1x768xf32>, tensor<1x1x1x768xf32>) -> tensor<401408x768xf32>
    %140 = stablehlo.dot_general %139, %arg66, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<401408x768xf32>, tensor<192x768xf32>) -> tensor<401408x192xf32>
    %141 = "mhlo.fusion"(%122, %140, %arg67) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<401408x192xf32>, %arg203: tensor<192xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<192xf32>) -> tensor<401408x192xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<401408x192xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<401408x192xf32>) -> tensor<512x28x28x192xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x28x28x192xf32>
      mhlo.return %401 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<401408x192xf32>, tensor<192xf32>) -> tensor<512x28x28x192xf32>
    %142 = "mhlo.fusion"(%141, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %143 = "mhlo.fusion"(%141, %142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<512x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %399 = stablehlo.multiply %arg202, %398 : tensor<512x28x28xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %401 = stablehlo.subtract %arg201, %400 : tensor<512x28x28x192xf32>
      %402 = stablehlo.multiply %401, %401 : tensor<512x28x28x192xf32>
      mhlo.return %402 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %144 = "mhlo.fusion"(%143, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28x192xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
      mhlo.return %398 : tensor<512x28x28x6xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x192xf32>, tensor<f32>) -> tensor<512x28x28x6xf32>
    %145 = "mhlo.fusion"(%144, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x28x28x6xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
      mhlo.return %398 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28x6xf32>, tensor<f32>) -> tensor<512x28x28xf32>
    %146 = "mhlo.fusion"(%145) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x28x28xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x28x28xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x28x28xf32>
      mhlo.return %402 : tensor<512x28x28xf32>
    }) {output_operand_aliasing = []} : (tensor<512x28x28xf32>) -> tensor<512x28x28xf32>
    %147 = "mhlo.fusion"(%arg69, %arg68, %146, %141, %142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<192xf32>, %arg202: tensor<192xf32>, %arg203: tensor<512x28x28xf32>, %arg204: tensor<512x28x28x192xf32>, %arg205: tensor<512x28x28xf32>):
      %cst_0 = stablehlo.constant dense<0.00520833349> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x28x28xf32>
      %399 = stablehlo.multiply %arg205, %398 : tensor<512x28x28xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %401 = stablehlo.subtract %arg204, %400 : tensor<512x28x28x192xf32>
      %402 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<512x28x28x192xf32>
      %404 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x28x28x192xf32>
      %406 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<192xf32>) -> tensor<512x28x28x192xf32>
      %407 = stablehlo.add %405, %406 : tensor<512x28x28x192xf32>
      mhlo.return %407 : tensor<512x28x28x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<512x28x28xf32>, tensor<512x28x28x192xf32>, tensor<512x28x28xf32>) -> tensor<512x28x28x192xf32>
    %148 = "mhlo.fusion"(%arg70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x192x2x2xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,192,384]{1,0,2,3}"} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
      %399 = mhlo.copy %398 : tensor<2x2x192x384xf32>
      mhlo.return %399 : tensor<2x2x192x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x192x2x2xf32>) -> tensor<2x2x192x384xf32>
    %149 = stablehlo.convolution(%147, %148) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x28x28x192xf32>, tensor<2x2x192x384xf32>) -> tensor<512x14x14x384xf32>
    %150 = "mhlo.fusion"(%149, %arg71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %151 = "mhlo.fusion"(%arg72) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %152 = stablehlo.convolution(%150, %151) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %153 = "mhlo.fusion"(%152, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %154 = "mhlo.fusion"(%153, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %155 = "mhlo.fusion"(%154, %152, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %156 = "mhlo.fusion"(%155, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %157 = "mhlo.fusion"(%156, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %158 = "mhlo.fusion"(%157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %159 = "mhlo.fusion"(%arg75, %arg74, %158, %154, %152, %arg73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %160 = stablehlo.dot_general %159, %arg76, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %161 = "mhlo.fusion"(%160, %arg77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %162 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %164 = "mhlo.fusion"(%163, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %165 = "mhlo.fusion"(%164, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %166 = "mhlo.fusion"(%165, %162) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %167 = "mhlo.fusion"(%161, %arg79, %166, %arg78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %168 = stablehlo.dot_general %167, %arg80, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %169 = "mhlo.fusion"(%168, %arg81, %149, %arg71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14x384xf32>, %arg204: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg203, %398 : tensor<512x14x14x384xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %401 = stablehlo.add %400, %arg201 : tensor<100352x384xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.add %399, %402 : tensor<512x14x14x384xf32>
      mhlo.return %403 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x384xf32>, tensor<384xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %170 = "mhlo.fusion"(%arg82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %171 = stablehlo.convolution(%169, %170) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %172 = "mhlo.fusion"(%171, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %173 = "mhlo.fusion"(%172, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %174 = "mhlo.fusion"(%173, %171, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %175 = "mhlo.fusion"(%174, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %176 = "mhlo.fusion"(%175, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %178 = "mhlo.fusion"(%arg85, %arg84, %177, %173, %171, %arg83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %179 = stablehlo.dot_general %178, %arg86, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %180 = "mhlo.fusion"(%179, %arg87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %181 = "mhlo.fusion"(%180) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %182 = "mhlo.fusion"(%181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %183 = "mhlo.fusion"(%182, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %184 = "mhlo.fusion"(%183, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %185 = "mhlo.fusion"(%184, %181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %186 = "mhlo.fusion"(%180, %arg89, %185, %arg88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %187 = stablehlo.dot_general %186, %arg90, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %188 = "mhlo.fusion"(%169, %187, %arg91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %189 = "mhlo.fusion"(%arg92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %190 = stablehlo.convolution(%188, %189) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %191 = "mhlo.fusion"(%190, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %192 = "mhlo.fusion"(%191, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %193 = "mhlo.fusion"(%192, %190, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %194 = "mhlo.fusion"(%193, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %195 = "mhlo.fusion"(%194, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %197 = "mhlo.fusion"(%arg95, %arg94, %196, %192, %190, %arg93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %198 = stablehlo.dot_general %197, %arg96, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %199 = "mhlo.fusion"(%198, %arg97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %200 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %202 = "mhlo.fusion"(%201, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %203 = "mhlo.fusion"(%202, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %204 = "mhlo.fusion"(%203, %200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %205 = "mhlo.fusion"(%199, %arg99, %204, %arg98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %206 = stablehlo.dot_general %205, %arg100, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %207 = "mhlo.fusion"(%188, %206, %arg101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %208 = "mhlo.fusion"(%arg102) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %209 = stablehlo.convolution(%207, %208) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %210 = "mhlo.fusion"(%209, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %211 = "mhlo.fusion"(%210, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %212 = "mhlo.fusion"(%211, %209, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %213 = "mhlo.fusion"(%212, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %214 = "mhlo.fusion"(%213, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %216 = "mhlo.fusion"(%arg105, %arg104, %215, %211, %209, %arg103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %217 = stablehlo.dot_general %216, %arg106, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %218 = "mhlo.fusion"(%217, %arg107) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %219 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %221 = "mhlo.fusion"(%220, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %222 = "mhlo.fusion"(%221, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %223 = "mhlo.fusion"(%222, %219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %224 = "mhlo.fusion"(%218, %arg109, %223, %arg108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %225 = stablehlo.dot_general %224, %arg110, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %226 = "mhlo.fusion"(%207, %225, %arg111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %227 = "mhlo.fusion"(%arg112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %228 = stablehlo.convolution(%226, %227) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %229 = "mhlo.fusion"(%228, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %230 = "mhlo.fusion"(%229, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %231 = "mhlo.fusion"(%230, %228, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %232 = "mhlo.fusion"(%231, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %233 = "mhlo.fusion"(%232, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %234 = "mhlo.fusion"(%233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %235 = "mhlo.fusion"(%arg115, %arg114, %234, %230, %228, %arg113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %236 = stablehlo.dot_general %235, %arg116, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %237 = "mhlo.fusion"(%236, %arg117) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %238 = "mhlo.fusion"(%237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %239 = "mhlo.fusion"(%238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %240 = "mhlo.fusion"(%239, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %241 = "mhlo.fusion"(%240, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %242 = "mhlo.fusion"(%241, %238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %243 = "mhlo.fusion"(%237, %arg119, %242, %arg118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %244 = stablehlo.dot_general %243, %arg120, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %245 = "mhlo.fusion"(%226, %244, %arg121) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %246 = "mhlo.fusion"(%arg122) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %247 = stablehlo.convolution(%245, %246) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %248 = "mhlo.fusion"(%247, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %249 = "mhlo.fusion"(%248, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %250 = "mhlo.fusion"(%249, %247, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %251 = "mhlo.fusion"(%250, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %252 = "mhlo.fusion"(%251, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %253 = "mhlo.fusion"(%252) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %254 = "mhlo.fusion"(%arg125, %arg124, %253, %249, %247, %arg123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %255 = stablehlo.dot_general %254, %arg126, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %256 = "mhlo.fusion"(%255, %arg127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %257 = "mhlo.fusion"(%256) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %259 = "mhlo.fusion"(%258, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %260 = "mhlo.fusion"(%259, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %261 = "mhlo.fusion"(%260, %257) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %262 = "mhlo.fusion"(%256, %arg129, %261, %arg128) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %263 = stablehlo.dot_general %262, %arg130, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %264 = "mhlo.fusion"(%245, %263, %arg131) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %265 = "mhlo.fusion"(%arg132) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %266 = stablehlo.convolution(%264, %265) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %267 = "mhlo.fusion"(%266, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %268 = "mhlo.fusion"(%267, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %269 = "mhlo.fusion"(%268, %266, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %270 = "mhlo.fusion"(%269, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %271 = "mhlo.fusion"(%270, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %272 = "mhlo.fusion"(%271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %273 = "mhlo.fusion"(%arg135, %arg134, %272, %268, %266, %arg133) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %274 = stablehlo.dot_general %273, %arg136, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %275 = "mhlo.fusion"(%274, %arg137) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %276 = "mhlo.fusion"(%275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %278 = "mhlo.fusion"(%277, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %279 = "mhlo.fusion"(%278, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %280 = "mhlo.fusion"(%279, %276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %281 = "mhlo.fusion"(%275, %arg139, %280, %arg138) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %282 = stablehlo.dot_general %281, %arg140, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %283 = "mhlo.fusion"(%264, %282, %arg141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %284 = "mhlo.fusion"(%arg142) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %285 = stablehlo.convolution(%283, %284) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %286 = "mhlo.fusion"(%285, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %287 = "mhlo.fusion"(%286, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %288 = "mhlo.fusion"(%287, %285, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %289 = "mhlo.fusion"(%288, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %290 = "mhlo.fusion"(%289, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %291 = "mhlo.fusion"(%290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %292 = "mhlo.fusion"(%arg145, %arg144, %291, %287, %285, %arg143) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %293 = stablehlo.dot_general %292, %arg146, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %294 = "mhlo.fusion"(%293, %arg147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %295 = "mhlo.fusion"(%294) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %296 = "mhlo.fusion"(%295) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %297 = "mhlo.fusion"(%296, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %298 = "mhlo.fusion"(%297, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %299 = "mhlo.fusion"(%298, %295) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %300 = "mhlo.fusion"(%294, %arg149, %299, %arg148) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %301 = stablehlo.dot_general %300, %arg150, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %302 = "mhlo.fusion"(%283, %301, %arg151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %303 = "mhlo.fusion"(%arg152) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,384]{1,0,2,3}"} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x384xf32>
      mhlo.return %399 : tensor<7x7x1x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384x1x7x7xf32>) -> tensor<7x7x1x384xf32>
    %304 = stablehlo.convolution(%302, %303) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 384 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<7x7x1x384xf32>) -> tensor<512x14x14x384xf32>
    %305 = "mhlo.fusion"(%304, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x14x14x384xf32>
      mhlo.return %399 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %306 = "mhlo.fusion"(%305, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %307 = "mhlo.fusion"(%306, %304, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>, %arg202: tensor<512x14x14x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x14x14x384xf32>
      mhlo.return %404 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %308 = "mhlo.fusion"(%307, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %309 = "mhlo.fusion"(%308, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %310 = "mhlo.fusion"(%309) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %311 = "mhlo.fusion"(%arg155, %arg154, %310, %306, %304, %arg153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14xf32>, %arg205: tensor<512x14x14x384xf32>, %arg206: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x14x14x384xf32>
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x14x14x384xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x14x14x384xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x384xf32>) -> tensor<100352x384xf32>
      mhlo.return %410 : tensor<100352x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<384xf32>) -> tensor<100352x384xf32>
    %312 = stablehlo.dot_general %311, %arg156, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x384xf32>, tensor<1536x384xf32>) -> tensor<100352x1536xf32>
    %313 = "mhlo.fusion"(%312, %arg157) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<100352x1536xf32>, %arg202: tensor<1536xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1536xf32>) -> tensor<100352x1536xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<100352x1536xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<100352x1536xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %403 = stablehlo.negate %400 : tensor<100352x1536xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<100352x1536xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<100352x1536xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x1536xf32>) -> tensor<512x14x14x1536xf32>
      %407 = stablehlo.abs %406 : tensor<512x14x14x1536xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x14x14x1536xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x14x14x1536xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x14x14x1536xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x14x14x1536xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x14x14x1536xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x14x14x1536xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x14x14x1536xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x14x14x1536xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x14x14x1536xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x14x14x1536xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x14x14x1536xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x14x14x1536xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x14x14x1536xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x14x14x1536xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x14x14x1536xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %435 = stablehlo.negate %411 : tensor<512x14x14x1536xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %438 = stablehlo.exponential %435 : tensor<512x14x14x1536xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x14x14x1536xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x14x14x1536xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x14x14x1536xf32>, tensor<512x14x14x1536xf32>) -> tensor<512x14x14x1536xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x14x14x1536xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x14x14x1536xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x14x14x1536xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x14x14x1536xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x14x14x1536xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x14x14x1536xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x14x14x1536xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x14x14x1536xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x14x14x1536xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x14x14x1536xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x14x14x1536xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x14x14x1536xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x14x14x1536xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x14x14x1536xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x14x14x1536xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x14x14x1536xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x14x14x1536xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x14x14x1536xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x14x14x1536xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x14x14x1536xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x14x14x1536xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x14x14x1536xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x14x14x1536xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x14x14x1536xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x14x14x1536xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x14x14x1536xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x14x14x1536xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x14x14x1536xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x14x14x1536xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x14x14x1536xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x14x14x1536xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x14x14x1536xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x14x14x1536xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x14x14x1536xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x14x14x1536xi1>, tensor<512x14x14x1536xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x14x14x1536xf32>
      mhlo.return %496 : tensor<512x14x14x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<100352x1536xf32>, tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
    %314 = "mhlo.fusion"(%313) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x14x14x1536xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x14x14x1536xf32>, tensor<f32>) -> tensor<512x1536xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x1536xf32>
      mhlo.return %400 : tensor<512x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>) -> tensor<512x1536xf32>
    %315 = "mhlo.fusion"(%314) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      mhlo.return %399 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %316 = "mhlo.fusion"(%315, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x1536xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [256, 256]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
      mhlo.return %398 : tensor<512x1x1x2xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x1536xf32>, tensor<f32>) -> tensor<512x1x1x2xf32>
    %317 = "mhlo.fusion"(%316, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x2xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x2xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %318 = "mhlo.fusion"(%317, %314) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x1536xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,1536]{3,0,2,1}"} : (tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x1536xf32>
      %cst_0 = stablehlo.constant dense<6.51041686E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x1536xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x1536xf32>
      mhlo.return %406 : tensor<512x1x1x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x1536xf32>) -> tensor<512x1x1x1536xf32>
    %319 = "mhlo.fusion"(%313, %arg159, %318, %arg158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x1536xf32>, %arg202: tensor<1x1x1x1536xf32>, %arg203: tensor<512x1x1x1536xf32>, %arg204: tensor<1x1x1x1536xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x1536xf32>) -> tensor<512x1536xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x1536xf32>) -> tensor<512x14x14x1536xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x14x14x1536xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x14x14x1536xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x1536xf32>) -> tensor<1536xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<1536xf32>) -> tensor<512x14x14x1536xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x14x14x1536xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x14x14x1536xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x14x14x1536xf32>) -> tensor<100352x1536xf32>
      mhlo.return %408 : tensor<100352x1536xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x1536xf32>, tensor<1x1x1x1536xf32>, tensor<512x1x1x1536xf32>, tensor<1x1x1x1536xf32>) -> tensor<100352x1536xf32>
    %320 = stablehlo.dot_general %319, %arg160, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100352x1536xf32>, tensor<384x1536xf32>) -> tensor<100352x384xf32>
    %321 = "mhlo.fusion"(%302, %320, %arg161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<100352x384xf32>, %arg203: tensor<384xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<384xf32>) -> tensor<100352x384xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<100352x384xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<100352x384xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x14x14x384xf32>
      mhlo.return %401 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<100352x384xf32>, tensor<384xf32>) -> tensor<512x14x14x384xf32>
    %322 = "mhlo.fusion"(%321, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %323 = "mhlo.fusion"(%321, %322) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg202, %398 : tensor<512x14x14xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.subtract %arg201, %400 : tensor<512x14x14x384xf32>
      %402 = stablehlo.multiply %401, %401 : tensor<512x14x14x384xf32>
      mhlo.return %402 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %324 = "mhlo.fusion"(%323, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14x384xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
      mhlo.return %398 : tensor<512x14x14x12xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x384xf32>, tensor<f32>) -> tensor<512x14x14x12xf32>
    %325 = "mhlo.fusion"(%324, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x14x14x12xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
      mhlo.return %398 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14x12xf32>, tensor<f32>) -> tensor<512x14x14xf32>
    %326 = "mhlo.fusion"(%325) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x14x14xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x14x14xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x14x14xf32>
      mhlo.return %402 : tensor<512x14x14xf32>
    }) {output_operand_aliasing = []} : (tensor<512x14x14xf32>) -> tensor<512x14x14xf32>
    %327 = "mhlo.fusion"(%arg163, %arg162, %326, %321, %322) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<384xf32>, %arg202: tensor<384xf32>, %arg203: tensor<512x14x14xf32>, %arg204: tensor<512x14x14x384xf32>, %arg205: tensor<512x14x14xf32>):
      %cst_0 = stablehlo.constant dense<0.00260416674> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x14x14xf32>
      %399 = stablehlo.multiply %arg205, %398 : tensor<512x14x14xf32>
      %400 = stablehlo.broadcast_in_dim %399, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %401 = stablehlo.subtract %arg204, %400 : tensor<512x14x14x384xf32>
      %402 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
      %403 = stablehlo.multiply %401, %402 : tensor<512x14x14x384xf32>
      %404 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x14x14x384xf32>
      %406 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<384xf32>) -> tensor<512x14x14x384xf32>
      %407 = stablehlo.add %405, %406 : tensor<512x14x14x384xf32>
      mhlo.return %407 : tensor<512x14x14x384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<512x14x14xf32>, tensor<512x14x14x384xf32>, tensor<512x14x14xf32>) -> tensor<512x14x14x384xf32>
    %328 = "mhlo.fusion"(%arg164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x384x2x2xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[2,2,384,768]{1,0,2,3}"} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
      %399 = mhlo.copy %398 : tensor<2x2x384x768xf32>
      mhlo.return %399 : tensor<2x2x384x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x384x2x2xf32>) -> tensor<2x2x384x768xf32>
    %329 = stablehlo.convolution(%327, %328) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x14x14x384xf32>, tensor<2x2x384x768xf32>) -> tensor<512x7x7x768xf32>
    %330 = "mhlo.fusion"(%329, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x7x7x768xf32>
      mhlo.return %399 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %331 = "mhlo.fusion"(%arg166) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x768xf32>
      mhlo.return %399 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %332 = stablehlo.convolution(%330, %331) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<512x7x7x768xf32>
    %333 = "mhlo.fusion"(%332, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x7x7x768xf32>
      mhlo.return %399 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %334 = "mhlo.fusion"(%333, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
      mhlo.return %398 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    %335 = "mhlo.fusion"(%334, %332, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7xf32>, %arg202: tensor<512x7x7x768xf32>, %arg203: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x7x7x768xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x7x7x768xf32>
      mhlo.return %404 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %336 = "mhlo.fusion"(%335, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
      mhlo.return %398 : tensor<512x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
    %337 = "mhlo.fusion"(%336, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
      mhlo.return %398 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    %338 = "mhlo.fusion"(%337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7xf32>):
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x7x7xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x7x7xf32>
      mhlo.return %402 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7xf32>) -> tensor<512x7x7xf32>
    %339 = "mhlo.fusion"(%arg169, %arg168, %338, %334, %332, %arg167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x7x7xf32>, %arg204: tensor<512x7x7xf32>, %arg205: tensor<512x7x7x768xf32>, %arg206: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x7x7x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x7x7x768xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x7x7x768xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x7x7x768xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x768xf32>) -> tensor<25088x768xf32>
      mhlo.return %410 : tensor<25088x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<512x7x7xf32>, tensor<512x7x7xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<25088x768xf32>
    %340 = stablehlo.dot_general %339, %arg170, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x768xf32>, tensor<3072x768xf32>) -> tensor<25088x3072xf32>
    %341 = "mhlo.fusion"(%340, %arg171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<25088x3072xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<25088x3072xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<25088x3072xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
      %403 = stablehlo.negate %400 : tensor<25088x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<25088x3072xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
      %407 = stablehlo.abs %406 : tensor<512x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x7x7x3072xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x7x7x3072xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x7x7x3072xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x7x7x3072xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x7x7x3072xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x7x7x3072xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x7x7x3072xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x7x7x3072xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %435 = stablehlo.negate %411 : tensor<512x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %438 = stablehlo.exponential %435 : tensor<512x7x7x3072xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x7x7x3072xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x7x7x3072xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x7x7x3072xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x7x7x3072xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x7x7x3072xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x7x7x3072xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x7x7x3072xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x7x7x3072xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x7x7x3072xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x7x7x3072xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x7x7x3072xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x7x7x3072xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x7x7x3072xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x7x7x3072xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x7x7x3072xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x7x7x3072xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x7x7x3072xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x7x7x3072xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x7x7x3072xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x7x7x3072xf32>
      mhlo.return %496 : tensor<512x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x3072xf32>, tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %342 = "mhlo.fusion"(%341) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x3072xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x3072xf32>, tensor<f32>) -> tensor<512x3072xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x3072xf32>
      mhlo.return %400 : tensor<512x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x3072xf32>) -> tensor<512x3072xf32>
    %343 = "mhlo.fusion"(%342) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x3072xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x3072xf32>
      mhlo.return %399 : tensor<512x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %344 = "mhlo.fusion"(%343, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x3072xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
      mhlo.return %398 : tensor<512x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
    %345 = "mhlo.fusion"(%344, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %346 = "mhlo.fusion"(%345, %342) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x3072xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x3072xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x3072xf32>
      mhlo.return %406 : tensor<512x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %347 = "mhlo.fusion"(%341, %arg173, %346, %arg172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<512x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x3072xf32>) -> tensor<512x3072xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x3072xf32>) -> tensor<512x7x7x3072xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x7x7x3072xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x7x7x3072xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x7x7x3072xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x7x7x3072xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x3072xf32>) -> tensor<25088x3072xf32>
      mhlo.return %408 : tensor<25088x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<512x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<25088x3072xf32>
    %348 = stablehlo.dot_general %347, %arg174, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x3072xf32>, tensor<768x3072xf32>) -> tensor<25088x768xf32>
    %349 = "mhlo.fusion"(%348, %arg175, %329, %arg165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x7x7x768xf32>, %arg204: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg204, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg203, %398 : tensor<512x7x7x768xf32>
      %400 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<25088x768xf32>
      %401 = stablehlo.add %400, %arg201 : tensor<25088x768xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x768xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.add %399, %402 : tensor<512x7x7x768xf32>
      mhlo.return %403 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x768xf32>, tensor<768xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %350 = "mhlo.fusion"(%arg176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x768xf32>
      mhlo.return %399 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %351 = stablehlo.convolution(%349, %350) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<512x7x7x768xf32>
    %352 = "mhlo.fusion"(%351, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x7x7x768xf32>
      mhlo.return %399 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %353 = "mhlo.fusion"(%352, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
      mhlo.return %398 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    %354 = "mhlo.fusion"(%353, %351, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7xf32>, %arg202: tensor<512x7x7x768xf32>, %arg203: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x7x7x768xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x7x7x768xf32>
      mhlo.return %404 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %355 = "mhlo.fusion"(%354, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
      mhlo.return %398 : tensor<512x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
    %356 = "mhlo.fusion"(%355, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
      mhlo.return %398 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    %357 = "mhlo.fusion"(%356) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7xf32>):
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x7x7xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x7x7xf32>
      mhlo.return %402 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7xf32>) -> tensor<512x7x7xf32>
    %358 = "mhlo.fusion"(%arg179, %arg178, %357, %353, %351, %arg177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x7x7xf32>, %arg204: tensor<512x7x7xf32>, %arg205: tensor<512x7x7x768xf32>, %arg206: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x7x7x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x7x7x768xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x7x7x768xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x7x7x768xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x768xf32>) -> tensor<25088x768xf32>
      mhlo.return %410 : tensor<25088x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<512x7x7xf32>, tensor<512x7x7xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<25088x768xf32>
    %359 = stablehlo.dot_general %358, %arg180, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x768xf32>, tensor<3072x768xf32>) -> tensor<25088x3072xf32>
    %360 = "mhlo.fusion"(%359, %arg181) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<25088x3072xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<25088x3072xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<25088x3072xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
      %403 = stablehlo.negate %400 : tensor<25088x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<25088x3072xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
      %407 = stablehlo.abs %406 : tensor<512x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x7x7x3072xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x7x7x3072xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x7x7x3072xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x7x7x3072xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x7x7x3072xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x7x7x3072xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x7x7x3072xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x7x7x3072xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %435 = stablehlo.negate %411 : tensor<512x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %438 = stablehlo.exponential %435 : tensor<512x7x7x3072xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x7x7x3072xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x7x7x3072xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x7x7x3072xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x7x7x3072xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x7x7x3072xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x7x7x3072xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x7x7x3072xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x7x7x3072xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x7x7x3072xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x7x7x3072xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x7x7x3072xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x7x7x3072xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x7x7x3072xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x7x7x3072xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x7x7x3072xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x7x7x3072xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x7x7x3072xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x7x7x3072xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x7x7x3072xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x7x7x3072xf32>
      mhlo.return %496 : tensor<512x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x3072xf32>, tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %361 = "mhlo.fusion"(%360) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x3072xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x3072xf32>, tensor<f32>) -> tensor<512x3072xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x3072xf32>
      mhlo.return %400 : tensor<512x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x3072xf32>) -> tensor<512x3072xf32>
    %362 = "mhlo.fusion"(%361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x3072xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x3072xf32>
      mhlo.return %399 : tensor<512x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %363 = "mhlo.fusion"(%362, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x3072xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
      mhlo.return %398 : tensor<512x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
    %364 = "mhlo.fusion"(%363, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %365 = "mhlo.fusion"(%364, %361) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x3072xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x3072xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x3072xf32>
      mhlo.return %406 : tensor<512x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %366 = "mhlo.fusion"(%360, %arg183, %365, %arg182) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<512x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x3072xf32>) -> tensor<512x3072xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x3072xf32>) -> tensor<512x7x7x3072xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x7x7x3072xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x7x7x3072xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x7x7x3072xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x7x7x3072xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x3072xf32>) -> tensor<25088x3072xf32>
      mhlo.return %408 : tensor<25088x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<512x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<25088x3072xf32>
    %367 = stablehlo.dot_general %366, %arg184, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x3072xf32>, tensor<768x3072xf32>) -> tensor<25088x768xf32>
    %368 = "mhlo.fusion"(%349, %367, %arg185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<25088x768xf32>, %arg203: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<25088x768xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<25088x768xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x768xf32>) -> tensor<512x7x7x768xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x7x7x768xf32>
      mhlo.return %401 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<25088x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %369 = "mhlo.fusion"(%arg186) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768x1x7x7xf32>):
      %398 = stablehlo.transpose %arg201, dims = [2, 3, 1, 0] {result_layout = dense<[1, 0, 2, 3]> : tensor<4xindex>, xla_shape = "f32[7,7,1,768]{1,0,2,3}"} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
      %399 = mhlo.copy %398 : tensor<7x7x1x768xf32>
      mhlo.return %399 : tensor<7x7x1x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768x1x7x7xf32>) -> tensor<7x7x1x768xf32>
    %370 = stablehlo.convolution(%368, %369) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 768 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<512x7x7x768xf32>, tensor<7x7x1x768xf32>) -> tensor<512x7x7x768xf32>
    %371 = "mhlo.fusion"(%370, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg201, %398 : tensor<512x7x7x768xf32>
      mhlo.return %399 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %372 = "mhlo.fusion"(%371, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
      mhlo.return %398 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    %373 = "mhlo.fusion"(%372, %370, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7xf32>, %arg202: tensor<512x7x7x768xf32>, %arg203: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg202, %398 : tensor<512x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x7x7x768xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x7x7x768xf32>
      mhlo.return %404 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %374 = "mhlo.fusion"(%373, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 32>, window_strides = array<i64: 1, 1, 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
      mhlo.return %398 : tensor<512x7x7x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x7x7x24xf32>
    %375 = "mhlo.fusion"(%374, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
      mhlo.return %398 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x24xf32>, tensor<f32>) -> tensor<512x7x7xf32>
    %376 = "mhlo.fusion"(%375) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7xf32>):
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x7x7xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.add %399, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.rsqrt %401 : tensor<512x7x7xf32>
      mhlo.return %402 : tensor<512x7x7xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7xf32>) -> tensor<512x7x7xf32>
    %377 = "mhlo.fusion"(%arg189, %arg188, %376, %372, %370, %arg187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512x7x7xf32>, %arg204: tensor<512x7x7xf32>, %arg205: tensor<512x7x7x768xf32>, %arg206: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg206, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %399 = stablehlo.add %arg205, %398 : tensor<512x7x7x768xf32>
      %cst_0 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x7x7xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512x7x7xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x7x7x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0, 1, 2] : (tensor<512x7x7xf32>) -> tensor<512x7x7x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x7x7x768xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x7x7x768xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [3] : (tensor<768xf32>) -> tensor<512x7x7x768xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x7x7x768xf32>
      %410 = mhlo.bitcast %409 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x768xf32>) -> tensor<25088x768xf32>
      mhlo.return %410 : tensor<25088x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<512x7x7xf32>, tensor<512x7x7xf32>, tensor<512x7x7x768xf32>, tensor<768xf32>) -> tensor<25088x768xf32>
    %378 = stablehlo.dot_general %377, %arg190, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x768xf32>, tensor<3072x768xf32>) -> tensor<25088x3072xf32>
    %379 = "mhlo.fusion"(%378, %arg191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<25088x3072xf32>, %arg202: tensor<3072xf32>):
      %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
      %399 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<3072xf32>) -> tensor<25088x3072xf32>
      %400 = stablehlo.add %399, %arg201 : tensor<25088x3072xf32>
      %401 = stablehlo.multiply %398, %400 : tensor<25088x3072xf32>
      %402 = mhlo.bitcast %401 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
      %403 = stablehlo.negate %400 : tensor<25088x3072xf32>
      %cst_1 = stablehlo.constant dense<0.707106769> : tensor<f32>
      %404 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<25088x3072xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<25088x3072xf32>
      %406 = mhlo.bitcast %405 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x3072xf32>) -> tensor<512x7x7x3072xf32>
      %407 = stablehlo.abs %406 : tensor<512x7x7x3072xf32>
      %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %408 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %409 = stablehlo.compare LT, %407, %408 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_3 = stablehlo.constant dense<7.85386146E-5> : tensor<f32>
      %410 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %411 = stablehlo.multiply %406, %406 : tensor<512x7x7x3072xf32>
      %412 = stablehlo.multiply %410, %411 : tensor<512x7x7x3072xf32>
      %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<f32>
      %413 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %414 = stablehlo.add %412, %413 : tensor<512x7x7x3072xf32>
      %415 = stablehlo.multiply %414, %411 : tensor<512x7x7x3072xf32>
      %cst_5 = stablehlo.constant dense<0.00518832775> : tensor<f32>
      %416 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %417 = stablehlo.add %415, %416 : tensor<512x7x7x3072xf32>
      %418 = stablehlo.multiply %417, %411 : tensor<512x7x7x3072xf32>
      %cst_6 = stablehlo.constant dense<-0.0268538129> : tensor<f32>
      %419 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %420 = stablehlo.add %418, %419 : tensor<512x7x7x3072xf32>
      %421 = stablehlo.multiply %420, %411 : tensor<512x7x7x3072xf32>
      %cst_7 = stablehlo.constant dense<0.112835854> : tensor<f32>
      %422 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %423 = stablehlo.add %421, %422 : tensor<512x7x7x3072xf32>
      %424 = stablehlo.multiply %423, %411 : tensor<512x7x7x3072xf32>
      %cst_8 = stablehlo.constant dense<-0.37612626> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %426 = stablehlo.add %424, %425 : tensor<512x7x7x3072xf32>
      %427 = stablehlo.multiply %426, %411 : tensor<512x7x7x3072xf32>
      %cst_9 = stablehlo.constant dense<1.12837911> : tensor<f32>
      %428 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %429 = stablehlo.add %427, %428 : tensor<512x7x7x3072xf32>
      %430 = stablehlo.multiply %406, %429 : tensor<512x7x7x3072xf32>
      %431 = stablehlo.subtract %408, %430 : tensor<512x7x7x3072xf32>
      %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %433 = stablehlo.compare LT, %406, %432 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_11 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %434 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %435 = stablehlo.negate %411 : tensor<512x7x7x3072xf32>
      %cst_12 = stablehlo.constant dense<-88.7228394> : tensor<f32>
      %436 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %437 = stablehlo.compare LT, %435, %436 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %438 = stablehlo.exponential %435 : tensor<512x7x7x3072xf32>
      %439 = stablehlo.divide %408, %407 : tensor<512x7x7x3072xf32>
      %440 = stablehlo.multiply %438, %439 : tensor<512x7x7x3072xf32>
      %441 = stablehlo.compare LT, %407, %434 : (tensor<512x7x7x3072xf32>, tensor<512x7x7x3072xf32>) -> tensor<512x7x7x3072xi1>
      %cst_13 = stablehlo.constant dense<2.326820e-02> : tensor<f32>
      %442 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %443 = stablehlo.divide %408, %411 : tensor<512x7x7x3072xf32>
      %444 = stablehlo.multiply %442, %443 : tensor<512x7x7x3072xf32>
      %cst_14 = stablehlo.constant dense<-0.138703942> : tensor<f32>
      %445 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %446 = stablehlo.add %444, %445 : tensor<512x7x7x3072xf32>
      %447 = stablehlo.multiply %446, %443 : tensor<512x7x7x3072xf32>
      %cst_15 = stablehlo.constant dense<0.368742466> : tensor<f32>
      %448 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %449 = stablehlo.add %447, %448 : tensor<512x7x7x3072xf32>
      %450 = stablehlo.multiply %449, %443 : tensor<512x7x7x3072xf32>
      %cst_16 = stablehlo.constant dense<-0.582473278> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %452 = stablehlo.add %450, %451 : tensor<512x7x7x3072xf32>
      %453 = stablehlo.multiply %452, %443 : tensor<512x7x7x3072xf32>
      %cst_17 = stablehlo.constant dense<0.621000468> : tensor<f32>
      %454 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %455 = stablehlo.add %453, %454 : tensor<512x7x7x3072xf32>
      %456 = stablehlo.multiply %455, %443 : tensor<512x7x7x3072xf32>
      %cst_18 = stablehlo.constant dense<-0.494451523> : tensor<f32>
      %457 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %458 = stablehlo.add %456, %457 : tensor<512x7x7x3072xf32>
      %459 = stablehlo.multiply %458, %443 : tensor<512x7x7x3072xf32>
      %cst_19 = stablehlo.constant dense<3.404880e-01> : tensor<f32>
      %460 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %461 = stablehlo.add %459, %460 : tensor<512x7x7x3072xf32>
      %462 = stablehlo.multiply %461, %443 : tensor<512x7x7x3072xf32>
      %cst_20 = stablehlo.constant dense<-0.274112701> : tensor<f32>
      %463 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %464 = stablehlo.add %462, %463 : tensor<512x7x7x3072xf32>
      %465 = stablehlo.multiply %464, %443 : tensor<512x7x7x3072xf32>
      %cst_21 = stablehlo.constant dense<0.563825965> : tensor<f32>
      %466 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %467 = stablehlo.add %465, %466 : tensor<512x7x7x3072xf32>
      %cst_22 = stablehlo.constant dense<-10.477664> : tensor<f32>
      %468 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %469 = stablehlo.multiply %468, %443 : tensor<512x7x7x3072xf32>
      %cst_23 = stablehlo.constant dense<1.297720e+01> : tensor<f32>
      %470 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %471 = stablehlo.add %469, %470 : tensor<512x7x7x3072xf32>
      %472 = stablehlo.multiply %471, %443 : tensor<512x7x7x3072xf32>
      %cst_24 = stablehlo.constant dense<-7.49551868> : tensor<f32>
      %473 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %474 = stablehlo.add %472, %473 : tensor<512x7x7x3072xf32>
      %475 = stablehlo.multiply %474, %443 : tensor<512x7x7x3072xf32>
      %cst_25 = stablehlo.constant dense<2.92101908> : tensor<f32>
      %476 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %477 = stablehlo.add %475, %476 : tensor<512x7x7x3072xf32>
      %478 = stablehlo.multiply %477, %443 : tensor<512x7x7x3072xf32>
      %cst_26 = stablehlo.constant dense<-1.01526523> : tensor<f32>
      %479 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %480 = stablehlo.add %478, %479 : tensor<512x7x7x3072xf32>
      %481 = stablehlo.multiply %480, %443 : tensor<512x7x7x3072xf32>
      %cst_27 = stablehlo.constant dense<0.42184633> : tensor<f32>
      %482 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %483 = stablehlo.add %481, %482 : tensor<512x7x7x3072xf32>
      %484 = stablehlo.multiply %483, %443 : tensor<512x7x7x3072xf32>
      %cst_28 = stablehlo.constant dense<-0.282076746> : tensor<f32>
      %485 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %486 = stablehlo.add %484, %485 : tensor<512x7x7x3072xf32>
      %487 = stablehlo.multiply %486, %443 : tensor<512x7x7x3072xf32>
      %cst_29 = stablehlo.constant dense<0.564189494> : tensor<f32>
      %488 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<512x7x7x3072xf32>
      %489 = stablehlo.add %487, %488 : tensor<512x7x7x3072xf32>
      %490 = stablehlo.select %441, %467, %489 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %491 = stablehlo.multiply %440, %490 : tensor<512x7x7x3072xf32>
      %492 = stablehlo.select %437, %432, %491 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %493 = stablehlo.subtract %434, %492 : tensor<512x7x7x3072xf32>
      %494 = stablehlo.select %433, %493, %492 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %495 = stablehlo.select %409, %431, %494 : tensor<512x7x7x3072xi1>, tensor<512x7x7x3072xf32>
      %496 = stablehlo.multiply %402, %495 : tensor<512x7x7x3072xf32>
      mhlo.return %496 : tensor<512x7x7x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<25088x3072xf32>, tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
    %380 = "mhlo.fusion"(%379) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x3072xf32>):
      %398 = stablehlo.multiply %arg201, %arg201 : tensor<512x7x7x3072xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %399 = stablehlo.reduce(%398 init: %cst_0) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x3072xf32>, tensor<f32>) -> tensor<512x3072xf32>
      %400 = stablehlo.sqrt %399 : tensor<512x3072xf32>
      mhlo.return %400 : tensor<512x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x3072xf32>) -> tensor<512x3072xf32>
    %381 = "mhlo.fusion"(%380) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x3072xf32>):
      %398 = mhlo.bitcast %arg201 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x3072xf32>
      mhlo.return %399 : tensor<512x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %382 = "mhlo.fusion"(%381, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x1x1x3072xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1024>, window_strides = array<i64: 1, 1, 1, 1024>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
      mhlo.return %398 : tensor<512x1x1x3xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x3072xf32>, tensor<f32>) -> tensor<512x1x1x3xf32>
    %383 = "mhlo.fusion"(%382, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1x3xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [3] : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
      mhlo.return %398 : tensor<512x1x1xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1x3xf32>, tensor<f32>) -> tensor<512x1x1xf32>
    %384 = "mhlo.fusion"(%383, %380) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1x1xf32>, %arg202: tensor<512x3072xf32>):
      %398 = mhlo.bitcast %arg202 {result_layout = dense<[3, 0, 2, 1]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>, xla_shape = "f32[512,1,1,3072]{3,0,2,1}"} : (tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
      %399 = mhlo.copy %398 : tensor<512x1x1x3072xf32>
      %cst_0 = stablehlo.constant dense<3.25520843E-4> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512x1x1xf32>
      %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
      %402 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x1x1xf32>
      %403 = stablehlo.add %401, %402 : tensor<512x1x1xf32>
      %404 = mhlo.bitcast %403 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<512x1x1xf32>) -> tensor<512xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [0] : (tensor<512xf32>) -> tensor<512x1x1x3072xf32>
      %406 = stablehlo.divide %399, %405 : tensor<512x1x1x3072xf32>
      mhlo.return %406 : tensor<512x1x1x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1x1xf32>, tensor<512x3072xf32>) -> tensor<512x1x1x3072xf32>
    %385 = "mhlo.fusion"(%379, %arg193, %384, %arg192) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x3072xf32>, %arg202: tensor<1x1x1x3072xf32>, %arg203: tensor<512x1x1x3072xf32>, %arg204: tensor<1x1x1x3072xf32>):
      %398 = mhlo.bitcast %arg204 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %399 = stablehlo.broadcast_in_dim %398, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
      %400 = mhlo.bitcast %arg203 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x1x1x3072xf32>) -> tensor<512x3072xf32>
      %401 = stablehlo.broadcast_in_dim %400, dims = [0, 3] : (tensor<512x3072xf32>) -> tensor<512x7x7x3072xf32>
      %402 = stablehlo.multiply %arg201, %401 : tensor<512x7x7x3072xf32>
      %403 = stablehlo.multiply %399, %402 : tensor<512x7x7x3072xf32>
      %404 = mhlo.bitcast %arg202 {result_layout = dense<0> : tensor<1xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x1x3072xf32>) -> tensor<3072xf32>
      %405 = stablehlo.broadcast_in_dim %404, dims = [3] : (tensor<3072xf32>) -> tensor<512x7x7x3072xf32>
      %406 = stablehlo.add %403, %405 : tensor<512x7x7x3072xf32>
      %407 = stablehlo.add %406, %arg201 : tensor<512x7x7x3072xf32>
      %408 = mhlo.bitcast %407 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<512x7x7x3072xf32>) -> tensor<25088x3072xf32>
      mhlo.return %408 : tensor<25088x3072xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x3072xf32>, tensor<1x1x1x3072xf32>, tensor<512x1x1x3072xf32>, tensor<1x1x1x3072xf32>) -> tensor<25088x3072xf32>
    %386 = stablehlo.dot_general %385, %arg194, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<25088x3072xf32>, tensor<768x3072xf32>) -> tensor<25088x768xf32>
    %387 = "mhlo.fusion"(%368, %386, %arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<25088x768xf32>, %arg203: tensor<768xf32>):
      %398 = stablehlo.broadcast_in_dim %arg203, dims = [1] : (tensor<768xf32>) -> tensor<25088x768xf32>
      %399 = stablehlo.add %398, %arg202 : tensor<25088x768xf32>
      %400 = mhlo.bitcast %399 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<25088x768xf32>) -> tensor<512x7x7x768xf32>
      %401 = stablehlo.add %arg201, %400 : tensor<512x7x7x768xf32>
      mhlo.return %401 : tensor<512x7x7x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<25088x768xf32>, tensor<768xf32>) -> tensor<512x7x7x768xf32>
    %388 = "mhlo.fusion"(%387, %cst) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg201: tensor<512x7x7x768xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1, 2] : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x768xf32>
      mhlo.return %398 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x7x7x768xf32>, tensor<f32>) -> tensor<512x768xf32>
    %389 = "mhlo.fusion"(%388) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x768xf32>
      %399 = stablehlo.multiply %arg201, %398 : tensor<512x768xf32>
      mhlo.return %399 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768xf32>) -> tensor<512x768xf32>
    %390 = "mhlo.fusion"(%389, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x768xf32>, tensor<f32>) -> tensor<512x24xf32>
      mhlo.return %398 : tensor<512x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768xf32>, tensor<f32>) -> tensor<512x24xf32>
    %391 = "mhlo.fusion"(%390, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x24xf32>, %arg202: tensor<f32>):
      %398 = stablehlo.reduce(%arg201 init: %arg202) applies stablehlo.add across dimensions = [1] : (tensor<512x24xf32>, tensor<f32>) -> tensor<512xf32>
      mhlo.return %398 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x24xf32>, tensor<f32>) -> tensor<512xf32>
    %392 = "mhlo.fusion"(%391, %388) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512xf32>, %arg202: tensor<512x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x768xf32>
      %399 = stablehlo.multiply %arg202, %398 : tensor<512x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %401 = stablehlo.multiply %arg201, %400 : tensor<512xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0] : (tensor<512xf32>) -> tensor<512x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x768xf32>
      %404 = stablehlo.multiply %403, %403 : tensor<512x768xf32>
      mhlo.return %404 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512x768xf32>) -> tensor<512x768xf32>
    %393 = "mhlo.fusion"(%392, %cst) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x768xf32>, %arg202: tensor<f32>):
      %398 = "stablehlo.reduce_window"(%arg201, %arg202) <{base_dilations = array<i64: 1, 1>, padding = dense<0> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 32>, window_strides = array<i64: 1, 32>}> ({
      ^bb0(%arg203: tensor<f32>, %arg204: tensor<f32>):
        %399 = stablehlo.add %arg203, %arg204 : tensor<f32>
        stablehlo.return %399 : tensor<f32>
      }) : (tensor<512x768xf32>, tensor<f32>) -> tensor<512x24xf32>
      mhlo.return %398 : tensor<512x24xf32>
    }) {output_operand_aliasing = []} : (tensor<512x768xf32>, tensor<f32>) -> tensor<512x24xf32>
    %394 = "mhlo.fusion"(%393) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x24xf32>):
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %398 = stablehlo.reduce(%arg201 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<512x24xf32>, tensor<f32>) -> tensor<512xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %399 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %400 = stablehlo.multiply %398, %399 : tensor<512xf32>
      %cst_2 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
      %401 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %402 = stablehlo.add %400, %401 : tensor<512xf32>
      %403 = stablehlo.rsqrt %402 : tensor<512xf32>
      mhlo.return %403 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512x24xf32>) -> tensor<512xf32>
    %395 = "mhlo.fusion"(%arg197, %arg196, %394, %391, %388) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<768xf32>, %arg202: tensor<768xf32>, %arg203: tensor<512xf32>, %arg204: tensor<512xf32>, %arg205: tensor<512x768xf32>):
      %cst_0 = stablehlo.constant dense<0.0204081628> : tensor<f32>
      %398 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x768xf32>
      %399 = stablehlo.multiply %arg205, %398 : tensor<512x768xf32>
      %cst_1 = stablehlo.constant dense<0.00130208337> : tensor<f32>
      %400 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %401 = stablehlo.multiply %arg204, %400 : tensor<512xf32>
      %402 = stablehlo.broadcast_in_dim %401, dims = [0] : (tensor<512xf32>) -> tensor<512x768xf32>
      %403 = stablehlo.subtract %399, %402 : tensor<512x768xf32>
      %404 = stablehlo.broadcast_in_dim %arg203, dims = [0] : (tensor<512xf32>) -> tensor<512x768xf32>
      %405 = stablehlo.multiply %403, %404 : tensor<512x768xf32>
      %406 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<768xf32>) -> tensor<512x768xf32>
      %407 = stablehlo.multiply %405, %406 : tensor<512x768xf32>
      %408 = stablehlo.broadcast_in_dim %arg201, dims = [1] : (tensor<768xf32>) -> tensor<512x768xf32>
      %409 = stablehlo.add %407, %408 : tensor<512x768xf32>
      mhlo.return %409 : tensor<512x768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x768xf32>) -> tensor<512x768xf32>
    %396 = stablehlo.dot_general %395, %arg198, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x768xf32>, tensor<1000x768xf32>) -> tensor<512x1000xf32>
    %397 = "mhlo.fusion"(%396, %arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg201: tensor<512x1000xf32>, %arg202: tensor<1000xf32>):
      %398 = stablehlo.broadcast_in_dim %arg202, dims = [1] : (tensor<1000xf32>) -> tensor<512x1000xf32>
      %399 = stablehlo.add %398, %arg201 : tensor<512x1000xf32>
      mhlo.return %399 : tensor<512x1000xf32>
    }) {output_operand_aliasing = []} : (tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<512x1000xf32>
    return %397 : tensor<512x1000xf32>
  }
}
