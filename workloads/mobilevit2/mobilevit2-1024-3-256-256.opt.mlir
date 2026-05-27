module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<1024x512xf32>, %arg1: tensor<2x512xf32>) -> tensor<1024x2xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024x512xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<2x512xf32>) -> tensor<1024x2xf32>
    return %2 : tensor<1024x2xf32>
  }
  func.func private @region_1.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<1024x192x4x64xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %1 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<1024x192x2x2x8x8xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x2x2x8x8xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x192x2x2x8x8xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x192x2x2x8x8xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x192x2x2x8x8xf32>) -> tensor<1024x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
    return %4 : tensor<1024x128xf32>
  }
  func.func private @region_1.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<1024x128x4x256xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %4 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %1 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<1024x128x4x256xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %4 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %1 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<1024x128x4x256xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %4 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %1 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<1024x128x4x256xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %4 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %1 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<1024x128x2x2x16x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x2x2x16x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x128x2x2x16x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x128x2x2x16x16xf32>) -> tensor<1024x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
    return %4 : tensor<1024x256xf32>
  }
  func.func private @region_1.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1024x512x8x8xf32>, %arg4: tensor<512xf32>) -> tensor<1024x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x512x8x8xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x512x8x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x512x8x8xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x512x8x8xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x512x8x8xf32>) -> tensor<1024x512x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x512x64xf32>, tensor<f32>) -> tensor<1024x512xf32>
    return %9 : tensor<1024x512xf32>
  }
  func.func private @region_1.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<1024x256x4x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<1024x256x4x16xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<1024x256x4x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<1024x256x4x16xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<1024x256x4x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<1024x256x4x16xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<1024x256x4x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<1024x256x4x16xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<1024x256x4x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<1024x256x4x16xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<1024x256x4x16xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<1024x256x4x16xf32>) -> tensor<1024xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<1024x256x2x2x4x4xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x2x2x4x4xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1024x256x2x2x4x4xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<1024x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x256x2x2x4x4xf32>) -> tensor<1024x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @region_1.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<1024x256x8x8xf32>) -> tensor<64x1024xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
    %8 = stablehlo.compare LT, %6, %7 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
    %10 = stablehlo.add %6, %9 : tensor<1x1x2x4xi32>
    %11 = stablehlo.select %8, %10, %6 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %15 = stablehlo.compare LT, %5, %14 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %17 = stablehlo.add %5, %16 : tensor<2x4xi32>
    %18 = stablehlo.select %15, %17, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %20 = stablehlo.concatenate %13, %19, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 256, 1, 1>}> : (tensor<1024x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1024x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1024x256x1x1xf32>) -> tensor<64x1024x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x256xf32>, tensor<f32>) -> tensor<64x1024xf32>
    return %24 : tensor<64x1024xf32>
  }
  func.func private @region_1.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<1024x192x16x16xf32>) -> tensor<256x1024xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
    %8 = stablehlo.compare LT, %6, %7 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
    %10 = stablehlo.add %6, %9 : tensor<1x1x2x8xi32>
    %11 = stablehlo.select %8, %10, %6 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %15 = stablehlo.compare LT, %5, %14 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %17 = stablehlo.add %5, %16 : tensor<2x8xi32>
    %18 = stablehlo.select %15, %17, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %20 = stablehlo.concatenate %13, %19, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 192, 1, 1>}> : (tensor<1024x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1024x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1024x192x1x1xf32>) -> tensor<256x1024x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x192xf32>, tensor<f32>) -> tensor<256x1024xf32>
    return %24 : tensor<256x1024xf32>
  }
  func.func private @region_1.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<1024x128x32x32xf32>) -> tensor<1024x1024xf32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
    %5 = stablehlo.compare LT, %3, %4 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
    %7 = stablehlo.add %3, %6 : tensor<1x1x2x16xi32>
    %8 = stablehlo.select %5, %7, %3 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %12 = stablehlo.compare LT, %2, %11 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %13 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %14 = stablehlo.add %2, %13 : tensor<2x16xi32>
    %15 = stablehlo.select %12, %14, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %17 = stablehlo.concatenate %10, %16, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 128, 1, 1>}> : (tensor<1024x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1024x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1024x128x1x1xf32>) -> tensor<1024x1024x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x1024x128xf32>, tensor<f32>) -> tensor<1024x1024xf32>
    return %21 : tensor<1024x1024xf32>
  }
  func.func private @region_1.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter(%arg0: tensor<1024x256x8x8xf32>, %arg1: tensor<2x4x2x4x1024x256xf32>) -> tensor<1024x256x8x8xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
    %8 = stablehlo.compare LT, %6, %7 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
    %10 = stablehlo.add %6, %9 : tensor<1x1x2x4xi32>
    %11 = stablehlo.select %8, %10, %6 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %15 = stablehlo.compare LT, %5, %14 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %17 = stablehlo.add %5, %16 : tensor<2x4xi32>
    %18 = stablehlo.select %15, %17, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %20 = stablehlo.concatenate %13, %19, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x1024x256xf32>) -> tensor<64x1024x256x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<1024x256x8x8xf32>, tensor<64x2xi32>, tensor<64x1024x256x1x1xf32>) -> tensor<1024x256x8x8xf32>
    return %23 : tensor<1024x256x8x8xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<256xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x4x16xf32>, %arg4: tensor<1024xf32>) -> tensor<2x4x2x4x1024x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x256x4x16xf32>
    %5 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x256x4x16xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x256x4x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x256x2x2x4x4xf32>
    %9 = stablehlo.transpose %8, dims = [2, 4, 3, 5, 0, 1] : (tensor<1024x256x2x2x4x4xf32>) -> tensor<2x4x2x4x1024x256xf32>
    return %9 : tensor<2x4x2x4x1024x256xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<1024x256x8x8xf32>) -> tensor<1024x256x2x2x4x4xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<4xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x4xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
    %8 = stablehlo.compare LT, %6, %7 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
    %10 = stablehlo.add %6, %9 : tensor<1x1x2x4xi32>
    %11 = stablehlo.select %8, %10, %6 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %15 = stablehlo.compare LT, %5, %14 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
    %17 = stablehlo.add %5, %16 : tensor<2x4xi32>
    %18 = stablehlo.select %15, %17, %5 : tensor<2x4xi1>, tensor<2x4xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
    %20 = stablehlo.concatenate %13, %19, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 256, 1, 1>}> : (tensor<1024x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1024x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1024x256x1x1xf32>) -> tensor<2x4x2x4x1024x256xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x1024x256xf32>) -> tensor<1024x256x2x2x4x4xf32>
    return %24 : tensor<1024x256x2x2x4x4xf32>
  }
  func.func private @region_1.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.1(%arg0: tensor<1024x192x16x16xf32>, %arg1: tensor<2x8x2x8x1024x192xf32>) -> tensor<1024x192x16x16xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
    %8 = stablehlo.compare LT, %6, %7 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
    %10 = stablehlo.add %6, %9 : tensor<1x1x2x8xi32>
    %11 = stablehlo.select %8, %10, %6 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %15 = stablehlo.compare LT, %5, %14 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %17 = stablehlo.add %5, %16 : tensor<2x8xi32>
    %18 = stablehlo.select %15, %17, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %20 = stablehlo.concatenate %13, %19, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x1024x192xf32>) -> tensor<256x1024x192x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<1024x192x16x16xf32>, tensor<256x2xi32>, tensor<256x1024x192x1x1xf32>) -> tensor<1024x192x16x16xf32>
    return %23 : tensor<1024x192x16x16xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<2x8x2x8x1024x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x192x2x2x8x8xf32>
    %9 = stablehlo.transpose %8, dims = [2, 4, 3, 5, 0, 1] : (tensor<1024x192x2x2x8x8xf32>) -> tensor<2x8x2x8x1024x192xf32>
    return %9 : tensor<2x8x2x8x1024x192xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<1024x192x16x16xf32>) -> tensor<1024x192x2x2x8x8xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
    %4 = stablehlo.iota dim = 0 : tensor<2x8xi32>
    %5 = stablehlo.add %3, %4 : tensor<2x8xi32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
    %8 = stablehlo.compare LT, %6, %7 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
    %c_1 = stablehlo.constant dense<16> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
    %10 = stablehlo.add %6, %9 : tensor<1x1x2x8xi32>
    %11 = stablehlo.select %8, %10, %6 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %15 = stablehlo.compare LT, %5, %14 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
    %17 = stablehlo.add %5, %16 : tensor<2x8xi32>
    %18 = stablehlo.select %15, %17, %5 : tensor<2x8xi1>, tensor<2x8xi32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
    %20 = stablehlo.concatenate %13, %19, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
    %21 = mhlo.bitcast %20 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 192, 1, 1>}> : (tensor<1024x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1024x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1024x192x1x1xf32>) -> tensor<2x8x2x8x1024x192xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x1024x192xf32>) -> tensor<1024x192x2x2x8x8xf32>
    return %24 : tensor<1024x192x2x2x8x8xf32>
  }
  func.func private @region_1.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.2(%arg0: tensor<1024x128x32x32xf32>, %arg1: tensor<2x16x2x16x1024x128xf32>) -> tensor<1024x128x32x32xf32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
    %5 = stablehlo.compare LT, %3, %4 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
    %7 = stablehlo.add %3, %6 : tensor<1x1x2x16xi32>
    %8 = stablehlo.select %5, %7, %3 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %12 = stablehlo.compare LT, %2, %11 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %13 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %14 = stablehlo.add %2, %13 : tensor<2x16xi32>
    %15 = stablehlo.select %12, %14, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %17 = stablehlo.concatenate %10, %16, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    %19 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x1024x128xf32>) -> tensor<1024x1024x128x1x1xf32>
    %20 = "stablehlo.scatter"(%arg0, %18, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %21 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }) : (tensor<1024x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x1024x128x1x1xf32>) -> tensor<1024x128x32x32xf32>
    return %20 : tensor<1024x128x32x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<1024xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x4x256xf32>, %arg4: tensor<1024xf32>) -> tensor<2x16x2x16x1024x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x128x4x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x128x2x2x16x16xf32>
    %9 = stablehlo.transpose %8, dims = [2, 4, 3, 5, 0, 1] : (tensor<1024x128x2x2x16x16xf32>) -> tensor<2x16x2x16x1024x128xf32>
    return %9 : tensor<2x16x2x16x1024x128xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<1024x128x32x32xf32>) -> tensor<1024x128x2x2x16x16xf32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x16xi32>
    %1 = stablehlo.iota dim = 0 : tensor<2x16xi32>
    %2 = stablehlo.add %0, %1 : tensor<2x16xi32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
    %5 = stablehlo.compare LT, %3, %4 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
    %c_0 = stablehlo.constant dense<32> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
    %7 = stablehlo.add %3, %6 : tensor<1x1x2x16xi32>
    %8 = stablehlo.select %5, %7, %3 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %12 = stablehlo.compare LT, %2, %11 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
    %13 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
    %14 = stablehlo.add %2, %13 : tensor<2x16xi32>
    %15 = stablehlo.select %12, %14, %2 : tensor<2x16xi1>, tensor<2x16xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
    %17 = stablehlo.concatenate %10, %16, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
    %18 = mhlo.bitcast %17 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 128, 1, 1>}> : (tensor<1024x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1024x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1024x128x1x1xf32>) -> tensor<2x16x2x16x1024x128xf32>
    %21 = stablehlo.transpose %20, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x1024x128xf32>) -> tensor<1024x128x2x2x16x16xf32>
    return %21 : tensor<1024x128x2x2x16x16xf32>
  }
  func.func private @fused_add(%arg0: tensor<1024xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x2x2x16x16xf32>, %arg4: tensor<1024xf32>) -> tensor<1x1024x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1x1024x131072xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x2x2x16x16xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<1024x128x2x2x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x2x2x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x128x2x2x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x128x2x2x16x16xf32>) -> tensor<1x1024x131072xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1x1024x131072xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x1024x131072xf32>
    %10 = stablehlo.add %1, %9 : tensor<1x1024x131072xf32>
    return %10 : tensor<1x1024x131072xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x2x2x8x8xf32>, %arg4: tensor<1024xf32>) -> tensor<1x1024x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1x1024x49152xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x2x2x8x8xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<1024x192x2x2x8x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x2x2x8x8xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x192x2x2x8x8xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x192x2x2x8x8xf32>) -> tensor<1x1024x49152xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1x1024x49152xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x1024x49152xf32>
    %10 = stablehlo.add %1, %9 : tensor<1x1024x49152xf32>
    return %10 : tensor<1x1024x49152xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x2x2x4x4xf32>, %arg4: tensor<1024xf32>) -> tensor<1x1024x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1x1024x16384xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x2x2x4x4xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<1024x256x2x2x4x4xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x2x2x4x4xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x2x2x4x4xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x256x2x2x4x4xf32>) -> tensor<1x1024x16384xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1x1024x16384xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x1024x16384xf32>
    %10 = stablehlo.add %1, %9 : tensor<1x1024x16384xf32>
    return %10 : tensor<1x1024x16384xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x128x128xf32>, %arg4: tensor<128xf32>) -> tensor<1024x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x128x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x128x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x128x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x128x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<1024x128x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x128x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x128x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x128x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x128x128x128xf32>
    return %13 : tensor<1024x128x128x128xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1024x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<1024x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x64x128x128xf32>
    return %7 : tensor<1024x64x128x128xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1024x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<1024x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<1024x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x64x128x128xf32>
    return %13 : tensor<1024x64x128x128xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1024x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<1024x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<1024x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x64x128x128xf32>
    return %13 : tensor<1024x64x128x128xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<1024x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x256x64x64xf32>
    return %13 : tensor<1024x256x64x64xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<1024x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x256x64x64xf32>
    return %13 : tensor<1024x256x64x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<1024x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x256x64x64xf32>
    return %13 : tensor<1024x256x64x64xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<1024x32x128x128xf32>, %arg4: tensor<32xf32>) -> tensor<1024x32x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x32x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x32x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x32x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x32x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x32x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<1024x32x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x32x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x32x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x32x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x32x128x128xf32>
    return %13 : tensor<1024x32x128x128xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<1024x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x128x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<1024x128x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x128x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x128x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x128x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x128x64x64xf32>
    return %13 : tensor<1024x128x64x64xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1024x512x32x32xf32>, %arg4: tensor<512xf32>) -> tensor<1024x512x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x512x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x512x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x512x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x512x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<1024x512x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x512x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x512x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x512x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x512x32x32xf32>
    return %13 : tensor<1024x512x32x32xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<1024x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x128x64x64xf32>
    return %7 : tensor<1024x128x64x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<1024x128x64x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<1024x128x64x64xf32>, %arg5: tensor<128xf32>) -> tensor<1024x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<1024x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x128x64x64xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<1024x128x64x64xf32>
    return %8 : tensor<1024x128x64x64xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x32x32xf32>
    return %7 : tensor<1024x256x32x32xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x32x32xf32>
    return %7 : tensor<1024x256x32x32xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<1024x256x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x256x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x256x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x256x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x256x32x32xf32>
    return %13 : tensor<1024x256x32x32xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<1024x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<1024x256x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x256x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x256x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x256x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x256x32x32xf32>
    return %13 : tensor<1024x256x32x32xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<1024x256x4x256xf32>) -> tensor<1024x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x256x4x256xf32>
    return %5 : tensor<1024x256x4x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<1024x256x4x256xf32>) -> tensor<1024x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x256x4x256xf32>
    return %5 : tensor<1024x256x4x256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<1024x768x16x16xf32>, %arg4: tensor<768xf32>) -> tensor<1024x768x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x768x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x768x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x768x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x768x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x768x16x16xf32>
    %9 = stablehlo.negate %7 : tensor<1024x768x16x16xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x768x16x16xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x768x16x16xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x768x16x16xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x768x16x16xf32>
    return %13 : tensor<1024x768x16x16xf32>
  }
  func.func private @region_1.1.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.198(%arg0: tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 129:257, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:129, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x128x4x256xf32>, tensor<f32>) -> tensor<1024x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x128x4xf32>) -> tensor<1024x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x128x4x256xf32>
    return %19 : tensor<1024x128x4x256xf32>
  }
  func.func private @region_1.1.clone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.199(%arg0: tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 129:257, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:129, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x128x4x256xf32>, tensor<f32>) -> tensor<1024x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x128x4xf32>) -> tensor<1024x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x128x4x256xf32>
    return %19 : tensor<1024x128x4x256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1024x512x16x16xf32>, %arg4: tensor<512xf32>) -> tensor<1024x512x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x512x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x512x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x512x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x512x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x16x16xf32>
    %9 = stablehlo.negate %7 : tensor<1024x512x16x16xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x512x16x16xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x512x16x16xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x512x16x16xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x512x16x16xf32>
    return %13 : tensor<1024x512x16x16xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<1024xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x4x256xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x128x4x256xf32>
    return %7 : tensor<1024x128x4x256xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<1024xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x4x256xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x128x4x256xf32>
    return %7 : tensor<1024x128x4x256xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<1024xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1024x128x4x256xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x128x4x256xf32>
    return %7 : tensor<1024x128x4x256xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1024x384x16x16xf32>, %arg4: tensor<384xf32>) -> tensor<1024x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x384x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x384x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x384x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x384x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x16x16xf32>
    %9 = stablehlo.negate %7 : tensor<1024x384x16x16xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x384x16x16xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x384x16x16xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x384x16x16xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x384x16x16xf32>
    return %13 : tensor<1024x384x16x16xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x384x4x64xf32>
    return %5 : tensor<1024x384x4x64xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x384x4x64xf32>
    return %5 : tensor<1024x384x4x64xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x384x4x64xf32>
    return %5 : tensor<1024x384x4x64xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x384x4x64xf32>
    return %5 : tensor<1024x384x4x64xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1024x384x16x16xf32>, %arg4: tensor<384xf32>) -> tensor<1024x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x384x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x384x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x384x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x384x16x16xf32>
    return %7 : tensor<1024x384x16x16xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<1024x384x16x16xf32>, %arg4: tensor<384xf32>) -> tensor<1024x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x384x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x384x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x384x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x384x16x16xf32>
    return %7 : tensor<1024x384x16x16xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<1024x768x8x8xf32>, %arg4: tensor<768xf32>) -> tensor<1024x768x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x768x8x8xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x768x8x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x768x8x8xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x768x8x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x768x8x8xf32>
    %9 = stablehlo.negate %7 : tensor<1024x768x8x8xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x768x8x8xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x768x8x8xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x768x8x8xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x768x8x8xf32>
    return %13 : tensor<1024x768x8x8xf32>
  }
  func.func private @region_1.1.clone.10(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.251(%arg0: tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x192x4x64xf32>
    return %19 : tensor<1024x192x4x64xf32>
  }
  func.func private @region_1.1.clone.8(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.252(%arg0: tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x192x4x64xf32>
    return %19 : tensor<1024x192x4x64xf32>
  }
  func.func private @region_1.1.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.253(%arg0: tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x192x4x64xf32>
    return %19 : tensor<1024x192x4x64xf32>
  }
  func.func private @region_1.1.clone.4(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.254(%arg0: tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x192x4x64xf32>
    return %19 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<1024xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<1024x192x4x64xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x192x4x64xf32>
    return %7 : tensor<1024x192x4x64xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1024x512x8x8xf32>, %arg4: tensor<512xf32>) -> tensor<1024x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x512x8x8xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x512x8x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x512x8x8xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x512x8x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x8x8xf32>
    %9 = stablehlo.negate %7 : tensor<1024x512x8x8xf32>
    %10 = stablehlo.exponential %9 : tensor<1024x512x8x8xf32>
    %11 = stablehlo.add %8, %10 : tensor<1024x512x8x8xf32>
    %12 = stablehlo.divide %8, %11 : tensor<1024x512x8x8xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<1024x512x8x8xf32>
    return %13 : tensor<1024x512x8x8xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<1024x512x4x16xf32>) -> tensor<1024x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x512x4x16xf32>
    return %5 : tensor<1024x512x4x16xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<1024x512x4x16xf32>) -> tensor<1024x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x512x4x16xf32>
    return %5 : tensor<1024x512x4x16xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<1024x512x4x16xf32>) -> tensor<1024x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<1024x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<1024x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<1024x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<1024x512x4x16xf32>
    return %5 : tensor<1024x512x4x16xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<1024x512x8x8xf32>, %arg4: tensor<512xf32>) -> tensor<1024x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<1024x512x8x8xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1024x512x8x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1024x512x8x8xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %7 = stablehlo.add %5, %6 : tensor<1024x512x8x8xf32>
    return %7 : tensor<1024x512x8x8xf32>
  }
  func.func private @region_1.1.clone.16(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.15(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.7(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.312(%arg0: tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 257:513, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:257, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x256x4x16xf32>, tensor<f32>) -> tensor<1024x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x256x4xf32>) -> tensor<1024x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x256x4x16xf32>
    return %19 : tensor<1024x256x4x16xf32>
  }
  func.func private @region_1.1.clone.14(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.13(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.6(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.313(%arg0: tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 257:513, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:257, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x256x4x16xf32>, tensor<f32>) -> tensor<1024x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x256x4xf32>) -> tensor<1024x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x256x4x16xf32>
    return %19 : tensor<1024x256x4x16xf32>
  }
  func.func private @region_1.1.clone.12(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_1.1.clone.11(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @region_5.9.clone.5(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_computation.314(%arg0: tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:1024, 257:513, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:1024, 1:257, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:1024, 0:1, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<1024x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<1024x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<1024x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<1024x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<1024x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x256x4x16xf32>, tensor<f32>) -> tensor<1024x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1024x256x4xf32>) -> tensor<1024x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<1024x256x4x16xf32>
    return %19 : tensor<1024x256x4x16xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x4x16xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x256x4x16xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x256x4x16xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x256x4x16xf32>
    return %7 : tensor<1024x256x4x16xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x4x16xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x256x4x16xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x256x4x16xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x256x4x16xf32>
    return %7 : tensor<1024x256x4x16xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x4x16xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x256x4x16xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x256x4x16xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x256x4x16xf32>
    return %7 : tensor<1024x256x4x16xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x4x16xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x256x4x16xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x256x4x16xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x256x4x16xf32>
    return %7 : tensor<1024x256x4x16xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<1024xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<1024x256x4x16xf32>, %arg4: tensor<1024xf32>) -> tensor<1024x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<1024x256x4x16xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1024x256x4x16xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<1024x256x4x16xf32>
    %7 = stablehlo.add %0, %6 : tensor<1024x256x4x16xf32>
    return %7 : tensor<1024x256x4x16xf32>
  }
  func.func private @fused_add.26(%arg0: tensor<1024x2xf32>, %arg1: tensor<2xf32>) -> tensor<1024x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<1024x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<1024x2xf32>
    return %1 : tensor<1024x2xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.36(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.37(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.38(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.39(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.40(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.41(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.42(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.43(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.44(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_multiply.45(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.3(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.4(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.5(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.6(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.7(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.8(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.9(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.10(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.11(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.12(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.13(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.14(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.15(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.16(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.17(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.18(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.19(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.20(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %4 = stablehlo.rsqrt %3 : tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
  func.func private @fused_rsqrt.21(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_rsqrt.22(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_rsqrt.23(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.24(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.25(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.26(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.27(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.28(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_rsqrt.29(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_rsqrt.30(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_rsqrt.31(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.32(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.33(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.34(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.35(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.36(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.37(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.38(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.39(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.40(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.41(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.42(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.43(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.44(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.45(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_broadcast() -> tensor<1024x192x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x16x16xf32>
    return %0 : tensor<1024x192x16x16xf32>
  }
  func.func private @fused_broadcast.1() -> tensor<1024x128x32x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x32x32xf32>
    return %0 : tensor<1024x128x32x32xf32>
  }
  func.func private @fused_broadcast.2() -> tensor<1024x256x8x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x8x8xf32>
    return %0 : tensor<1024x256x8x8xf32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<1024x1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.52(%arg0: tensor<1024x256xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.53(%arg0: tensor<256x1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.54(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.55(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.56(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.57(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.58(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.59(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.60(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.61(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.62(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.63(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.64(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.65(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.66(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.67(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.68(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.69(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.70(%arg0: tensor<1024x128xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func private @fused_reduce.71(%arg0: tensor<64x1024xf32>) -> tensor<1024xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x1024xf32>, tensor<f32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<1024x3x256x256xf32>) -> tensor<1024x2xf32> {
    %0 = "mhlo.fusion"(%arg243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<512xf32>
      %349 = stablehlo.rsqrt %348 : tensor<512xf32>
      mhlo.return %349 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x8x8xf32>
      mhlo.return %347 : tensor<1024x256x8x8xf32>
    }) {output_operand_aliasing = []} : () -> tensor<1024x256x8x8xf32>
    %2 = "mhlo.fusion"(%arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<512xf32>
      %349 = stablehlo.rsqrt %348 : tensor<512xf32>
      mhlo.return %349 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %3 = "mhlo.fusion"(%arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<512xf32>
      %349 = stablehlo.rsqrt %348 : tensor<512xf32>
      mhlo.return %349 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %4 = "mhlo.fusion"(%arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<768xf32>
      %349 = stablehlo.rsqrt %348 : tensor<768xf32>
      mhlo.return %349 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %5 = "mhlo.fusion"(%arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<768xf32>
      %349 = stablehlo.rsqrt %348 : tensor<768xf32>
      mhlo.return %349 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %6 = "mhlo.fusion"(%arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<384xf32>
      %349 = stablehlo.rsqrt %348 : tensor<384xf32>
      mhlo.return %349 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %7 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x16x16xf32>
      mhlo.return %347 : tensor<1024x192x16x16xf32>
    }) {output_operand_aliasing = []} : () -> tensor<1024x192x16x16xf32>
    %8 = "mhlo.fusion"(%arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<384xf32>
      %349 = stablehlo.rsqrt %348 : tensor<384xf32>
      mhlo.return %349 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %9 = "mhlo.fusion"(%arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<384xf32>
      %349 = stablehlo.rsqrt %348 : tensor<384xf32>
      mhlo.return %349 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %10 = "mhlo.fusion"(%arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<512xf32>
      %349 = stablehlo.rsqrt %348 : tensor<512xf32>
      mhlo.return %349 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %11 = "mhlo.fusion"(%arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<512xf32>
      %349 = stablehlo.rsqrt %348 : tensor<512xf32>
      mhlo.return %349 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %12 = "mhlo.fusion"(%arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x32x32xf32>
      mhlo.return %347 : tensor<1024x128x32x32xf32>
    }) {output_operand_aliasing = []} : () -> tensor<1024x128x32x32xf32>
    %14 = "mhlo.fusion"(%arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %15 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %16 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %17 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %18 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<128xf32>
      %349 = stablehlo.rsqrt %348 : tensor<128xf32>
      mhlo.return %349 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %19 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<128xf32>
      %349 = stablehlo.rsqrt %348 : tensor<128xf32>
      mhlo.return %349 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %20 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<128xf32>
      %349 = stablehlo.rsqrt %348 : tensor<128xf32>
      mhlo.return %349 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %21 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<64xf32>
      %349 = stablehlo.rsqrt %348 : tensor<64xf32>
      mhlo.return %349 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %22 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<64xf32>
      %349 = stablehlo.rsqrt %348 : tensor<64xf32>
      mhlo.return %349 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %23 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<64xf32>
      %349 = stablehlo.rsqrt %348 : tensor<64xf32>
      mhlo.return %349 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %24 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<32xf32>
      %349 = stablehlo.rsqrt %348 : tensor<32xf32>
      mhlo.return %349 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %25 = stablehlo.custom_call @__cudnn$convForward(%arg244, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x3x256x256xf32>, tensor<32x3x3x3xf32>) -> tuple<tensor<1024x32x128x128xf32>, tensor<0xui8>>
    %26 = stablehlo.get_tuple_element %25[0] : (tuple<tensor<1024x32x128x128xf32>, tensor<0xui8>>) -> tensor<1024x32x128x128xf32>
    %27 = "mhlo.fusion"(%arg2, %arg1, %24, %26, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<1024x32x128x128xf32>, %arg249: tensor<32xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x32x128x128xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x32x128x128xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x32x128x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<32xf32>) -> tensor<1024x32x128x128xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x32x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x32x128x128xf32>
      %356 = stablehlo.negate %354 : tensor<1024x32x128x128xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x32x128x128xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x32x128x128xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x32x128x128xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x32x128x128xf32>
      mhlo.return %360 : tensor<1024x32x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<1024x32x128x128xf32>, tensor<32xf32>) -> tensor<1024x32x128x128xf32>
    %28 = stablehlo.custom_call @__cudnn$convForward(%27, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x32x128x128xf32>, tensor<64x32x1x1xf32>) -> tuple<tensor<1024x64x128x128xf32>, tensor<0xui8>>
    %29 = stablehlo.get_tuple_element %28[0] : (tuple<tensor<1024x64x128x128xf32>, tensor<0xui8>>) -> tensor<1024x64x128x128xf32>
    %30 = "mhlo.fusion"(%arg5, %arg4, %23, %29, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<1024x64x128x128xf32>, %arg249: tensor<64xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x64x128x128xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x64x128x128xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x64x128x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x64x128x128xf32>
      %356 = stablehlo.negate %354 : tensor<1024x64x128x128xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x64x128x128xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x64x128x128xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x64x128x128xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x64x128x128xf32>
      mhlo.return %360 : tensor<1024x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1024x64x128x128xf32>, tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %31 = stablehlo.custom_call @__cudnn$convForward(%30, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x64x128x128xf32>, tensor<64x1x3x3xf32>) -> tuple<tensor<1024x64x128x128xf32>, tensor<0xui8>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<1024x64x128x128xf32>, tensor<0xui8>>) -> tensor<1024x64x128x128xf32>
    %33 = "mhlo.fusion"(%arg8, %arg7, %22, %32, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<1024x64x128x128xf32>, %arg249: tensor<64xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x64x128x128xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x64x128x128xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x64x128x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x64x128x128xf32>
      %356 = stablehlo.negate %354 : tensor<1024x64x128x128xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x64x128x128xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x64x128x128xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x64x128x128xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x64x128x128xf32>
      mhlo.return %360 : tensor<1024x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1024x64x128x128xf32>, tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %34 = stablehlo.custom_call @__cudnn$convForward(%33, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x64x128x128xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<1024x64x128x128xf32>, tensor<0xui8>>
    %35 = stablehlo.get_tuple_element %34[0] : (tuple<tensor<1024x64x128x128xf32>, tensor<0xui8>>) -> tensor<1024x64x128x128xf32>
    %36 = "mhlo.fusion"(%arg11, %arg10, %21, %35, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<1024x64x128x128xf32>, %arg249: tensor<64xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x64x128x128xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x64x128x128xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x64x128x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<1024x64x128x128xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x64x128x128xf32>
      mhlo.return %354 : tensor<1024x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1024x64x128x128xf32>, tensor<64xf32>) -> tensor<1024x64x128x128xf32>
    %37 = stablehlo.custom_call @__cudnn$convForward(%36, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x64x128x128xf32>, tensor<128x64x1x1xf32>) -> tuple<tensor<1024x128x128x128xf32>, tensor<0xui8>>
    %38 = stablehlo.get_tuple_element %37[0] : (tuple<tensor<1024x128x128x128xf32>, tensor<0xui8>>) -> tensor<1024x128x128x128xf32>
    %39 = "mhlo.fusion"(%arg14, %arg13, %20, %38, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x128x128xf32>, %arg249: tensor<128xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x128x128x128xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x128x128x128xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x128x128x128xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x128x128xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x128x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x128x128xf32>
      %356 = stablehlo.negate %354 : tensor<1024x128x128x128xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x128x128x128xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x128x128x128xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x128x128x128xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x128x128x128xf32>
      mhlo.return %360 : tensor<1024x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x128x128xf32>, tensor<128xf32>) -> tensor<1024x128x128x128xf32>
    %40 = stablehlo.custom_call @__cudnn$convForward(%39, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<1024x128x64x64xf32>, tensor<0xui8>>
    %41 = stablehlo.get_tuple_element %40[0] : (tuple<tensor<1024x128x64x64xf32>, tensor<0xui8>>) -> tensor<1024x128x64x64xf32>
    %42 = "mhlo.fusion"(%arg17, %arg16, %19, %41, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x64x64xf32>, %arg249: tensor<128xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x128x64x64xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x128x64x64xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x128x64x64xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x128x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x64x64xf32>
      %356 = stablehlo.negate %354 : tensor<1024x128x64x64xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x128x64x64xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x128x64x64xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x128x64x64xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x128x64x64xf32>
      mhlo.return %360 : tensor<1024x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x64x64xf32>, tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %43 = stablehlo.custom_call @__cudnn$convForward(%42, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x64x64xf32>, tensor<128x128x1x1xf32>) -> tuple<tensor<1024x128x64x64xf32>, tensor<0xui8>>
    %44 = stablehlo.get_tuple_element %43[0] : (tuple<tensor<1024x128x64x64xf32>, tensor<0xui8>>) -> tensor<1024x128x64x64xf32>
    %45 = "mhlo.fusion"(%arg20, %arg19, %18, %44, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x64x64xf32>, %arg249: tensor<128xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x128x64x64xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x128x64x64xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x128x64x64xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x128x64x64xf32>
      mhlo.return %354 : tensor<1024x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x64x64xf32>, tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %46 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<128xf32>
      %349 = stablehlo.rsqrt %348 : tensor<128xf32>
      mhlo.return %349 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %47 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %48 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %348 = stablehlo.add %arg245, %347 : tensor<256xf32>
      %349 = stablehlo.rsqrt %348 : tensor<256xf32>
      mhlo.return %349 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %49 = stablehlo.custom_call @__cudnn$convForward(%45, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<1024x256x64x64xf32>, tensor<0xui8>>
    %50 = stablehlo.get_tuple_element %49[0] : (tuple<tensor<1024x256x64x64xf32>, tensor<0xui8>>) -> tensor<1024x256x64x64xf32>
    %51 = "mhlo.fusion"(%arg23, %arg22, %48, %50, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x64x64xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x64x64xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x64x64xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x64x64xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x64x64xf32>
      %356 = stablehlo.negate %354 : tensor<1024x256x64x64xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x256x64x64xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x256x64x64xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x256x64x64xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x256x64x64xf32>
      mhlo.return %360 : tensor<1024x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x64x64xf32>, tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %52 = stablehlo.custom_call @__cudnn$convForward(%51, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1024x256x64x64xf32>, tensor<0xui8>>
    %53 = stablehlo.get_tuple_element %52[0] : (tuple<tensor<1024x256x64x64xf32>, tensor<0xui8>>) -> tensor<1024x256x64x64xf32>
    %54 = "mhlo.fusion"(%arg26, %arg25, %47, %53, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x64x64xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x64x64xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x64x64xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x64x64xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x64x64xf32>
      %356 = stablehlo.negate %354 : tensor<1024x256x64x64xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x256x64x64xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x256x64x64xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x256x64x64xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x256x64x64xf32>
      mhlo.return %360 : tensor<1024x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x64x64xf32>, tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %55 = stablehlo.custom_call @__cudnn$convForward(%54, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<1024x128x64x64xf32>, tensor<0xui8>>
    %56 = stablehlo.get_tuple_element %55[0] : (tuple<tensor<1024x128x64x64xf32>, tensor<0xui8>>) -> tensor<1024x128x64x64xf32>
    %57 = "mhlo.fusion"(%45, %arg29, %arg28, %46, %56, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x128x64x64xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<128xf32>, %arg249: tensor<1024x128x64x64xf32>, %arg250: tensor<128xf32>):
      %347 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %348 = stablehlo.subtract %arg249, %347 : tensor<1024x128x64x64xf32>
      %349 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x128x64x64xf32>
      %351 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x128x64x64xf32>
      %353 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x64x64xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x128x64x64xf32>
      %355 = stablehlo.add %arg245, %354 : tensor<1024x128x64x64xf32>
      mhlo.return %355 : tensor<1024x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x64x64xf32>, tensor<128xf32>) -> tensor<1024x128x64x64xf32>
    %58 = stablehlo.custom_call @__cudnn$convForward(%57, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<1024x256x64x64xf32>, tensor<0xui8>>
    %59 = stablehlo.get_tuple_element %58[0] : (tuple<tensor<1024x256x64x64xf32>, tensor<0xui8>>) -> tensor<1024x256x64x64xf32>
    %60 = "mhlo.fusion"(%arg32, %arg31, %17, %59, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x64x64xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x64x64xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x64x64xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x64x64xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x64x64xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x64x64xf32>
      %356 = stablehlo.negate %354 : tensor<1024x256x64x64xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x256x64x64xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x256x64x64xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x256x64x64xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x256x64x64xf32>
      mhlo.return %360 : tensor<1024x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x64x64xf32>, tensor<256xf32>) -> tensor<1024x256x64x64xf32>
    %61 = stablehlo.custom_call @__cudnn$convForward(%60, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>
    %62 = stablehlo.get_tuple_element %61[0] : (tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>) -> tensor<1024x256x32x32xf32>
    %63 = "mhlo.fusion"(%arg35, %arg34, %16, %62, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x32x32xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x32x32xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x32x32xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x32x32xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x32x32xf32>
      %356 = stablehlo.negate %354 : tensor<1024x256x32x32xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x256x32x32xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x256x32x32xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x256x32x32xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x256x32x32xf32>
      mhlo.return %360 : tensor<1024x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x32x32xf32>, tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %64 = stablehlo.custom_call @__cudnn$convForward(%63, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x32x32xf32>, tensor<256x256x1x1xf32>) -> tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>
    %65 = stablehlo.get_tuple_element %64[0] : (tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>) -> tensor<1024x256x32x32xf32>
    %66 = "mhlo.fusion"(%arg38, %arg37, %15, %65, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x32x32xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x32x32xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x32x32xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x32x32xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x32x32xf32>
      mhlo.return %354 : tensor<1024x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x32x32xf32>, tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %67 = stablehlo.custom_call @__cudnn$convForward(%66, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x32x32xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>
    %68 = stablehlo.get_tuple_element %67[0] : (tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>) -> tensor<1024x256x32x32xf32>
    %69 = "mhlo.fusion"(%arg41, %arg40, %14, %68, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x32x32xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x32x32xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x32x32xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x32x32xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x32x32xf32>
      %356 = stablehlo.negate %354 : tensor<1024x256x32x32xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x256x32x32xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x256x32x32xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x256x32x32xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x256x32x32xf32>
      mhlo.return %360 : tensor<1024x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x32x32xf32>, tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %70 = stablehlo.custom_call @__cudnn$convForward(%69, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x32x32xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<1024x128x32x32xf32>, tensor<0xui8>>
    %71 = stablehlo.get_tuple_element %70[0] : (tuple<tensor<1024x128x32x32xf32>, tensor<0xui8>>) -> tensor<1024x128x32x32xf32>
    %72 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x32x32xf32>):
      %347 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %348 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %349 = stablehlo.add %347, %348 : tensor<2x16xi32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %351 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %352 = stablehlo.compare LT, %350, %351 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %353 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %354 = stablehlo.add %350, %353 : tensor<1x1x2x16xi32>
      %355 = stablehlo.select %352, %354, %350 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %356 = mhlo.bitcast %355 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %358 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %359 = stablehlo.compare LT, %349, %358 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %360 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %361 = stablehlo.add %349, %360 : tensor<2x16xi32>
      %362 = stablehlo.select %359, %361, %349 : tensor<2x16xi1>, tensor<2x16xi32>
      %363 = stablehlo.broadcast_in_dim %362, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %364 = stablehlo.concatenate %357, %363, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %365 = mhlo.bitcast %364 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %366 = "stablehlo.gather"(%arg245, %365) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 128, 1, 1>}> : (tensor<1024x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1024x128x1x1xf32>
      %367 = mhlo.bitcast %366 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1024x128x1x1xf32>) -> tensor<2x16x2x16x1024x128xf32>
      %368 = stablehlo.transpose %367, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x1024x128xf32>) -> tensor<1024x128x2x2x16x16xf32>
      mhlo.return %368 : tensor<1024x128x2x2x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x32x32xf32>) -> tensor<1024x128x2x2x16x16xf32>
    %73 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x32x32xf32>):
      %347 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %348 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %349 = stablehlo.add %347, %348 : tensor<2x16xi32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %351 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %352 = stablehlo.compare LT, %350, %351 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %353 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %354 = stablehlo.add %350, %353 : tensor<1x1x2x16xi32>
      %355 = stablehlo.select %352, %354, %350 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %356 = mhlo.bitcast %355 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %358 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %359 = stablehlo.compare LT, %349, %358 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %360 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %361 = stablehlo.add %349, %360 : tensor<2x16xi32>
      %362 = stablehlo.select %359, %361, %349 : tensor<2x16xi1>, tensor<2x16xi32>
      %363 = stablehlo.broadcast_in_dim %362, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %364 = stablehlo.concatenate %357, %363, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %365 = mhlo.bitcast %364 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %366 = "stablehlo.gather"(%arg245, %365) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 128, 1, 1>}> : (tensor<1024x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x1024x128x1x1xf32>
      %367 = mhlo.bitcast %366 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x1024x128x1x1xf32>) -> tensor<1024x1024x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %368 = stablehlo.reduce(%367 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x1024x128xf32>, tensor<f32>) -> tensor<1024x1024xf32>
      mhlo.return %368 : tensor<1024x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x32x32xf32>) -> tensor<1024x1024xf32>
    %74 = "mhlo.fusion"(%73) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x1024xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x1024xf32>) -> tensor<1024xf32>
    %75 = "mhlo.fusion"(%74) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %76 = "mhlo.fusion"(%72, %75) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x2x2x16x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x2x2x16x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x128x2x2x16x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x128x2x2x16x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x128x2x2x16x16xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %351 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x2x2x16x16xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %77 = "mhlo.fusion"(%76) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %78 = "mhlo.fusion"(%77) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %79 = "mhlo.fusion"(%78, %arg43, %arg44, %72, %75) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x2x2x16x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1x1024x131072xf32>
      %349 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x2x2x16x16xf32>
      %350 = stablehlo.subtract %arg248, %349 : tensor<1024x128x2x2x16x16xf32>
      %351 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x2x2x16x16xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x128x2x2x16x16xf32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x128x2x2x16x16xf32>) -> tensor<1x1024x131072xf32>
      %354 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1x1024x131072xf32>
      %356 = stablehlo.multiply %353, %355 : tensor<1x1024x131072xf32>
      %357 = stablehlo.add %348, %356 : tensor<1x1024x131072xf32>
      mhlo.return %357 : tensor<1x1024x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x2x2x16x16xf32>, tensor<1024xf32>) -> tensor<1x1024x131072xf32>
    %80 = mhlo.bitcast %79 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x131072xf32>) -> tensor<1024x128x4x256xf32>
    %81 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%80, %arg45, %arg46) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<1024x257x4x256xf32>, tensor<0xui8>>
    %82 = stablehlo.get_tuple_element %81[0] : (tuple<tensor<1024x257x4x256xf32>, tensor<0xui8>>) -> tensor<1024x257x4x256xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x257x4x256xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 129:257, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x4x256xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x128x4x256xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:129, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x1x4x256xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x256xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x256xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x256xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x256xf32>) -> tensor<1024x128x4x256xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x128x4x256xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x128x4x256xf32>, tensor<f32>) -> tensor<1024x128x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x128x4xf32>) -> tensor<1024x128x4x256xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x128x4x256xf32>
      mhlo.return %366 : tensor<1024x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %84 = mhlo.bitcast %72 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x128x2x2x16x16xf32>) -> tensor<1024x128x4x256xf32>
    %85 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%83, %arg47, %arg48, %84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>) -> tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>
    %86 = stablehlo.get_tuple_element %85[0] : (tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>) -> tensor<1024x128x4x256xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %348 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32>
    %88 = "mhlo.fusion"(%87) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %89 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %90 = "mhlo.fusion"(%86, %89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x128x4x256xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x128x4x256xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %351 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %91 = "mhlo.fusion"(%90) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %92 = "mhlo.fusion"(%91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %93 = "mhlo.fusion"(%92, %arg49, %arg50, %86, %89) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x4x256xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x128x4x256xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x128x4x256xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x128x4x256xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x128x4x256xf32>
      mhlo.return %354 : tensor<1024x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %94 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%93, %arg51, %arg52) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<1024x256x4x256xf32>, tensor<0xui8>>
    %95 = stablehlo.get_tuple_element %94[0] : (tuple<tensor<1024x256x4x256xf32>, tensor<0xui8>>) -> tensor<1024x256x4x256xf32>
    %96 = "mhlo.fusion"(%95) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x256xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x256x4x256xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x256x4x256xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x256x4x256xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x256x4x256xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x256x4x256xf32>
      mhlo.return %352 : tensor<1024x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x256xf32>) -> tensor<1024x256x4x256xf32>
    %97 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%96, %arg53, %arg54, %86) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>) -> tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>
    %98 = stablehlo.get_tuple_element %97[0] : (tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>) -> tensor<1024x128x4x256xf32>
    %99 = "mhlo.fusion"(%98) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %348 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32>
    %100 = "mhlo.fusion"(%99) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %101 = "mhlo.fusion"(%100) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %102 = "mhlo.fusion"(%98, %101) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x128x4x256xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x128x4x256xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %351 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %103 = "mhlo.fusion"(%102) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %105 = "mhlo.fusion"(%104, %arg55, %arg56, %98, %101) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x4x256xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x128x4x256xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x128x4x256xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x128x4x256xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x128x4x256xf32>
      mhlo.return %354 : tensor<1024x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %106 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%105, %arg57, %arg58) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<1024x257x4x256xf32>, tensor<0xui8>>
    %107 = stablehlo.get_tuple_element %106[0] : (tuple<tensor<1024x257x4x256xf32>, tensor<0xui8>>) -> tensor<1024x257x4x256xf32>
    %108 = "mhlo.fusion"(%107) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x257x4x256xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 129:257, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x128x4x256xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x128x4x256xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:129, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:256] : (tensor<1024x257x4x256xf32>) -> tensor<1024x1x4x256xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x256xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x256xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x256xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x256xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x256xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x256xf32>) -> tensor<1024x4x256xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x256xf32>) -> tensor<1024x128x4x256xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x128x4x256xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x128x4x256xf32>, tensor<f32>) -> tensor<1024x128x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x128x4xf32>) -> tensor<1024x128x4x256xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x128x4x256xf32>
      mhlo.return %366 : tensor<1024x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x257x4x256xf32>) -> tensor<1024x128x4x256xf32>
    %109 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%108, %arg59, %arg60, %98) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>) -> tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>
    %110 = stablehlo.get_tuple_element %109[0] : (tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>) -> tensor<1024x128x4x256xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %348 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32>
    %112 = "mhlo.fusion"(%111) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %113 = "mhlo.fusion"(%112) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %114 = "mhlo.fusion"(%110, %113) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x128x4x256xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x128x4x256xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %351 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %117 = "mhlo.fusion"(%116, %arg61, %arg62, %110, %113) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x4x256xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x128x4x256xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x128x4x256xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x128x4x256xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x128x4x256xf32>
      mhlo.return %354 : tensor<1024x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
    %118 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%117, %arg63, %arg64) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<1024x256x4x256xf32>, tensor<0xui8>>
    %119 = stablehlo.get_tuple_element %118[0] : (tuple<tensor<1024x256x4x256xf32>, tensor<0xui8>>) -> tensor<1024x256x4x256xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x256xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x256x4x256xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x256x4x256xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x256x4x256xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x256x4x256xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x256x4x256xf32>
      mhlo.return %352 : tensor<1024x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x256xf32>) -> tensor<1024x256x4x256xf32>
    %121 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%120, %arg65, %arg66, %110) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>) -> tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>
    %122 = stablehlo.get_tuple_element %121[0] : (tuple<tensor<1024x128x4x256xf32>, tensor<0xui8>>) -> tensor<1024x128x4x256xf32>
    %123 = "mhlo.fusion"(%122) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %348 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256xf32>
    %124 = "mhlo.fusion"(%123) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %125 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %126 = "mhlo.fusion"(%122, %125) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x4x256xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x128x4x256xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x128x4x256xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x256x512xf32>, tensor<f32>) -> tensor<1024x256xf32>
      mhlo.return %351 : tensor<1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<1024x256xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256xf32>) -> tensor<1024xf32>
    %128 = "mhlo.fusion"(%127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %129 = "mhlo.fusion"(%128, %arg67, %arg68, %122, %125) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<1024x128x4x256xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x128x4x256xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x128x4x256xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x128x4x256xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<1024x128x4x256xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x128x4x256xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x128x4x256xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x128x4x256xf32>) -> tensor<1024x128x2x2x16x16xf32>
      %356 = stablehlo.transpose %355, dims = [2, 4, 3, 5, 0, 1] : (tensor<1024x128x2x2x16x16xf32>) -> tensor<2x16x2x16x1024x128xf32>
      mhlo.return %356 : tensor<2x16x2x16x1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1024x128x4x256xf32>, tensor<1024xf32>) -> tensor<2x16x2x16x1024x128xf32>
    %130 = "mhlo.fusion"(%13, %129) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128x32x32xf32>, %arg246: tensor<2x16x2x16x1024x128xf32>):
      %347 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %348 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %349 = stablehlo.add %347, %348 : tensor<2x16xi32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %351 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %352 = stablehlo.compare LT, %350, %351 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %353 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %354 = stablehlo.add %350, %353 : tensor<1x1x2x16xi32>
      %355 = stablehlo.select %352, %354, %350 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %356 = mhlo.bitcast %355 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %358 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %359 = stablehlo.compare LT, %349, %358 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %360 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %361 = stablehlo.add %349, %360 : tensor<2x16xi32>
      %362 = stablehlo.select %359, %361, %349 : tensor<2x16xi1>, tensor<2x16xi32>
      %363 = stablehlo.broadcast_in_dim %362, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %364 = stablehlo.concatenate %357, %363, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %365 = mhlo.bitcast %364 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %366 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x1024x128xf32>) -> tensor<1024x1024x128x1x1xf32>
      %367 = "stablehlo.scatter"(%arg245, %365, %366) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %368 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %368 : tensor<f32>
      }) : (tensor<1024x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x1024x128x1x1xf32>) -> tensor<1024x128x32x32xf32>
      mhlo.return %367 : tensor<1024x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128x32x32xf32>, tensor<2x16x2x16x1024x128xf32>) -> tensor<1024x128x32x32xf32>
    %131 = stablehlo.custom_call @__cudnn$convForward(%130, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x128x32x32xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>
    %132 = stablehlo.get_tuple_element %131[0] : (tuple<tensor<1024x256x32x32xf32>, tensor<0xui8>>) -> tensor<1024x256x32x32xf32>
    %133 = "mhlo.fusion"(%arg71, %arg70, %12, %132, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x32x32xf32>, %arg249: tensor<256xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x256x32x32xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x256x32x32xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x32x32xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x32x32xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x256x32x32xf32>
      mhlo.return %354 : tensor<1024x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x32x32xf32>, tensor<256xf32>) -> tensor<1024x256x32x32xf32>
    %134 = stablehlo.custom_call @__cudnn$convForward(%133, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x32x32xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<1024x512x32x32xf32>, tensor<0xui8>>
    %135 = stablehlo.get_tuple_element %134[0] : (tuple<tensor<1024x512x32x32xf32>, tensor<0xui8>>) -> tensor<1024x512x32x32xf32>
    %136 = "mhlo.fusion"(%arg74, %arg73, %11, %135, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1024x512x32x32xf32>, %arg249: tensor<512xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x512x32x32xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x512x32x32xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x512x32x32xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x32x32xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x512x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x32x32xf32>
      %356 = stablehlo.negate %354 : tensor<1024x512x32x32xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x512x32x32xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x512x32x32xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x512x32x32xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x512x32x32xf32>
      mhlo.return %360 : tensor<1024x512x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1024x512x32x32xf32>, tensor<512xf32>) -> tensor<1024x512x32x32xf32>
    %137 = stablehlo.custom_call @__cudnn$convForward(%136, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x32x32xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<1024x512x16x16xf32>, tensor<0xui8>>
    %138 = stablehlo.get_tuple_element %137[0] : (tuple<tensor<1024x512x16x16xf32>, tensor<0xui8>>) -> tensor<1024x512x16x16xf32>
    %139 = "mhlo.fusion"(%arg77, %arg76, %10, %138, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1024x512x16x16xf32>, %arg249: tensor<512xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x512x16x16xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x512x16x16xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x512x16x16xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x16x16xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x512x16x16xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x16x16xf32>
      %356 = stablehlo.negate %354 : tensor<1024x512x16x16xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x512x16x16xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x512x16x16xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x512x16x16xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x512x16x16xf32>
      mhlo.return %360 : tensor<1024x512x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1024x512x16x16xf32>, tensor<512xf32>) -> tensor<1024x512x16x16xf32>
    %140 = stablehlo.custom_call @__cudnn$convForward(%139, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x16x16xf32>, tensor<384x512x1x1xf32>) -> tuple<tensor<1024x384x16x16xf32>, tensor<0xui8>>
    %141 = stablehlo.get_tuple_element %140[0] : (tuple<tensor<1024x384x16x16xf32>, tensor<0xui8>>) -> tensor<1024x384x16x16xf32>
    %142 = "mhlo.fusion"(%9, %arg80, %arg79, %141, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<1024x384x16x16xf32>, %arg249: tensor<384xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x384x16x16xf32>
      %349 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x384x16x16xf32>
      %351 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x384x16x16xf32>
      %353 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x384x16x16xf32>
      mhlo.return %354 : tensor<1024x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1024x384x16x16xf32>, tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %143 = stablehlo.custom_call @__cudnn$convForward(%142, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x16x16xf32>, tensor<384x1x3x3xf32>) -> tuple<tensor<1024x384x16x16xf32>, tensor<0xui8>>
    %144 = stablehlo.get_tuple_element %143[0] : (tuple<tensor<1024x384x16x16xf32>, tensor<0xui8>>) -> tensor<1024x384x16x16xf32>
    %145 = "mhlo.fusion"(%arg83, %arg82, %8, %144, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<1024x384x16x16xf32>, %arg249: tensor<384xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x384x16x16xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x384x16x16xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x384x16x16xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x384x16x16xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x16x16xf32>
      %356 = stablehlo.negate %354 : tensor<1024x384x16x16xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x384x16x16xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x384x16x16xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x384x16x16xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x384x16x16xf32>
      mhlo.return %360 : tensor<1024x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1024x384x16x16xf32>, tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %146 = stablehlo.custom_call @__cudnn$convForward(%145, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x16x16xf32>, tensor<192x384x1x1xf32>) -> tuple<tensor<1024x192x16x16xf32>, tensor<0xui8>>
    %147 = stablehlo.get_tuple_element %146[0] : (tuple<tensor<1024x192x16x16xf32>, tensor<0xui8>>) -> tensor<1024x192x16x16xf32>
    %148 = "mhlo.fusion"(%147) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x16x16xf32>):
      %347 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %349 = stablehlo.multiply %347, %348 : tensor<8xi32>
      %350 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %351 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %352 = stablehlo.add %350, %351 : tensor<2x8xi32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %354 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %355 = stablehlo.compare LT, %353, %354 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %356 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %357 = stablehlo.add %353, %356 : tensor<1x1x2x8xi32>
      %358 = stablehlo.select %355, %357, %353 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %361 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %362 = stablehlo.compare LT, %352, %361 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %363 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %364 = stablehlo.add %352, %363 : tensor<2x8xi32>
      %365 = stablehlo.select %362, %364, %352 : tensor<2x8xi1>, tensor<2x8xi32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %367 = stablehlo.concatenate %360, %366, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %369 = "stablehlo.gather"(%arg245, %368) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 192, 1, 1>}> : (tensor<1024x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1024x192x1x1xf32>
      %370 = mhlo.bitcast %369 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1024x192x1x1xf32>) -> tensor<2x8x2x8x1024x192xf32>
      %371 = stablehlo.transpose %370, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x1024x192xf32>) -> tensor<1024x192x2x2x8x8xf32>
      mhlo.return %371 : tensor<1024x192x2x2x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x16x16xf32>) -> tensor<1024x192x2x2x8x8xf32>
    %149 = "mhlo.fusion"(%147) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x16x16xf32>):
      %347 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %349 = stablehlo.multiply %347, %348 : tensor<8xi32>
      %350 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %351 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %352 = stablehlo.add %350, %351 : tensor<2x8xi32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %354 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %355 = stablehlo.compare LT, %353, %354 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %356 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %357 = stablehlo.add %353, %356 : tensor<1x1x2x8xi32>
      %358 = stablehlo.select %355, %357, %353 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %361 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %362 = stablehlo.compare LT, %352, %361 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %363 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %364 = stablehlo.add %352, %363 : tensor<2x8xi32>
      %365 = stablehlo.select %362, %364, %352 : tensor<2x8xi1>, tensor<2x8xi32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %367 = stablehlo.concatenate %360, %366, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %369 = "stablehlo.gather"(%arg245, %368) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 192, 1, 1>}> : (tensor<1024x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x1024x192x1x1xf32>
      %370 = mhlo.bitcast %369 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x1024x192x1x1xf32>) -> tensor<256x1024x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %371 = stablehlo.reduce(%370 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x1024x192xf32>, tensor<f32>) -> tensor<256x1024xf32>
      mhlo.return %371 : tensor<256x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x16x16xf32>) -> tensor<256x1024xf32>
    %150 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256x1024xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<256x1024xf32>) -> tensor<1024xf32>
    %151 = "mhlo.fusion"(%150) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %152 = "mhlo.fusion"(%148, %151) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x2x2x8x8xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x2x2x8x8xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x2x2x8x8xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x2x2x8x8xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x192x2x2x8x8xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x2x2x8x8xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %153 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %154 = "mhlo.fusion"(%153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %155 = "mhlo.fusion"(%154, %arg85, %arg86, %148, %151) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x2x2x8x8xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1x1024x49152xf32>
      %349 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x2x2x8x8xf32>
      %350 = stablehlo.subtract %arg248, %349 : tensor<1024x192x2x2x8x8xf32>
      %351 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x2x2x8x8xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x192x2x2x8x8xf32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x192x2x2x8x8xf32>) -> tensor<1x1024x49152xf32>
      %354 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1x1024x49152xf32>
      %356 = stablehlo.multiply %353, %355 : tensor<1x1024x49152xf32>
      %357 = stablehlo.add %348, %356 : tensor<1x1024x49152xf32>
      mhlo.return %357 : tensor<1x1024x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x2x2x8x8xf32>, tensor<1024xf32>) -> tensor<1x1024x49152xf32>
    %156 = mhlo.bitcast %155 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x49152xf32>) -> tensor<1024x192x4x64xf32>
    %157 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%156, %arg87, %arg88) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>
    %158 = stablehlo.get_tuple_element %157[0] : (tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>) -> tensor<1024x385x4x64xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x385x4x64xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x64xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x64xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x64xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x192x4x64xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x192x4x64xf32>
      mhlo.return %366 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %160 = mhlo.bitcast %148 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x192x2x2x8x8xf32>) -> tensor<1024x192x4x64xf32>
    %161 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%159, %arg89, %arg90, %160) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %162 = stablehlo.get_tuple_element %161[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %166 = "mhlo.fusion"(%162, %165) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %167 = "mhlo.fusion"(%166) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %168 = "mhlo.fusion"(%167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %169 = "mhlo.fusion"(%168, %arg91, %arg92, %162, %165) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %170 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%169, %arg93, %arg94) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>
    %171 = stablehlo.get_tuple_element %170[0] : (tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>) -> tensor<1024x384x4x64xf32>
    %172 = "mhlo.fusion"(%171) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x384x4x64xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x384x4x64xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x384x4x64xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x384x4x64xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x384x4x64xf32>
      mhlo.return %352 : tensor<1024x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32>
    %173 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%172, %arg95, %arg96, %162) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %174 = stablehlo.get_tuple_element %173[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %176 = "mhlo.fusion"(%175) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %177 = "mhlo.fusion"(%176) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %178 = "mhlo.fusion"(%174, %177) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %180 = "mhlo.fusion"(%179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %181 = "mhlo.fusion"(%180, %arg97, %arg98, %174, %177) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %182 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%181, %arg99, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>
    %183 = stablehlo.get_tuple_element %182[0] : (tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>) -> tensor<1024x385x4x64xf32>
    %184 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x385x4x64xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x64xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x64xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x64xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x192x4x64xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x192x4x64xf32>
      mhlo.return %366 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %185 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%184, %arg101, %arg102, %174) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %186 = stablehlo.get_tuple_element %185[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %188 = "mhlo.fusion"(%187) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %190 = "mhlo.fusion"(%186, %189) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %192 = "mhlo.fusion"(%191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %193 = "mhlo.fusion"(%192, %arg103, %arg104, %186, %189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %194 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%193, %arg105, %arg106) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>
    %195 = stablehlo.get_tuple_element %194[0] : (tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>) -> tensor<1024x384x4x64xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x384x4x64xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x384x4x64xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x384x4x64xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x384x4x64xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x384x4x64xf32>
      mhlo.return %352 : tensor<1024x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32>
    %197 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%196, %arg107, %arg108, %186) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %198 = stablehlo.get_tuple_element %197[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %200 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %202 = "mhlo.fusion"(%198, %201) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %203 = "mhlo.fusion"(%202) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %204 = "mhlo.fusion"(%203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %205 = "mhlo.fusion"(%204, %arg109, %arg110, %198, %201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %206 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%205, %arg111, %arg112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>
    %207 = stablehlo.get_tuple_element %206[0] : (tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>) -> tensor<1024x385x4x64xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x385x4x64xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x64xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x64xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x64xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x192x4x64xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x192x4x64xf32>
      mhlo.return %366 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %209 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%208, %arg113, %arg114, %198) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %210 = stablehlo.get_tuple_element %209[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %212 = "mhlo.fusion"(%211) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %213 = "mhlo.fusion"(%212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %214 = "mhlo.fusion"(%210, %213) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %216 = "mhlo.fusion"(%215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %217 = "mhlo.fusion"(%216, %arg115, %arg116, %210, %213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %218 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%217, %arg117, %arg118) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>
    %219 = stablehlo.get_tuple_element %218[0] : (tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>) -> tensor<1024x384x4x64xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x384x4x64xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x384x4x64xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x384x4x64xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x384x4x64xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x384x4x64xf32>
      mhlo.return %352 : tensor<1024x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32>
    %221 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%220, %arg119, %arg120, %210) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %222 = stablehlo.get_tuple_element %221[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %223 = "mhlo.fusion"(%222) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %224 = "mhlo.fusion"(%223) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %226 = "mhlo.fusion"(%222, %225) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %227 = "mhlo.fusion"(%226) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %228 = "mhlo.fusion"(%227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %229 = "mhlo.fusion"(%228, %arg121, %arg122, %222, %225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %230 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%229, %arg123, %arg124) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>
    %231 = stablehlo.get_tuple_element %230[0] : (tuple<tensor<1024x385x4x64xf32>, tensor<0xui8>>) -> tensor<1024x385x4x64xf32>
    %232 = "mhlo.fusion"(%231) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x385x4x64xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 193:385, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:193, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:64] : (tensor<1024x385x4x64xf32>) -> tensor<1024x1x4x64xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x64xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x64xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x64xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x64xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x64xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x64xf32>) -> tensor<1024x4x64xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x64xf32>) -> tensor<1024x192x4x64xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x192x4x64xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x192x4x64xf32>, tensor<f32>) -> tensor<1024x192x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x192x4xf32>) -> tensor<1024x192x4x64xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x192x4x64xf32>
      mhlo.return %366 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x385x4x64xf32>) -> tensor<1024x192x4x64xf32>
    %233 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%232, %arg125, %arg126, %222) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %234 = stablehlo.get_tuple_element %233[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %236 = "mhlo.fusion"(%235) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %237 = "mhlo.fusion"(%236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %238 = "mhlo.fusion"(%234, %237) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %239 = "mhlo.fusion"(%238) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %240 = "mhlo.fusion"(%239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %241 = "mhlo.fusion"(%240, %arg127, %arg128, %234, %237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      mhlo.return %354 : tensor<1024x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
    %242 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%241, %arg129, %arg130) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>
    %243 = stablehlo.get_tuple_element %242[0] : (tuple<tensor<1024x384x4x64xf32>, tensor<0xui8>>) -> tensor<1024x384x4x64xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x384x4x64xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x384x4x64xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x384x4x64xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x384x4x64xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x384x4x64xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x384x4x64xf32>
      mhlo.return %352 : tensor<1024x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x384x4x64xf32>) -> tensor<1024x384x4x64xf32>
    %245 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%244, %arg131, %arg132, %234) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>) -> tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>
    %246 = stablehlo.get_tuple_element %245[0] : (tuple<tensor<1024x192x4x64xf32>, tensor<0xui8>>) -> tensor<1024x192x4x64xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %348 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128xf32>
    %248 = "mhlo.fusion"(%247) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %249 = "mhlo.fusion"(%248) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %250 = "mhlo.fusion"(%246, %249) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x4x64xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x192x4x64xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x192x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x128x384xf32>, tensor<f32>) -> tensor<1024x128xf32>
      mhlo.return %351 : tensor<1024x128xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<1024x128xf32>
    %251 = "mhlo.fusion"(%250) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x128xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x128xf32>) -> tensor<1024xf32>
    %252 = "mhlo.fusion"(%251) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %253 = "mhlo.fusion"(%252, %arg133, %arg134, %246, %249) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<1024x192x4x64xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x192x4x64xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x192x4x64xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x192x4x64xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<1024x192x4x64xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x192x4x64xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x192x4x64xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x192x4x64xf32>) -> tensor<1024x192x2x2x8x8xf32>
      %356 = stablehlo.transpose %355, dims = [2, 4, 3, 5, 0, 1] : (tensor<1024x192x2x2x8x8xf32>) -> tensor<2x8x2x8x1024x192xf32>
      mhlo.return %356 : tensor<2x8x2x8x1024x192xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<192xf32>, tensor<192xf32>, tensor<1024x192x4x64xf32>, tensor<1024xf32>) -> tensor<2x8x2x8x1024x192xf32>
    %254 = "mhlo.fusion"(%7, %253) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x192x16x16xf32>, %arg246: tensor<2x8x2x8x1024x192xf32>):
      %347 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %349 = stablehlo.multiply %347, %348 : tensor<8xi32>
      %350 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %351 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %352 = stablehlo.add %350, %351 : tensor<2x8xi32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %354 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %355 = stablehlo.compare LT, %353, %354 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %356 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %357 = stablehlo.add %353, %356 : tensor<1x1x2x8xi32>
      %358 = stablehlo.select %355, %357, %353 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %361 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %362 = stablehlo.compare LT, %352, %361 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %363 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %364 = stablehlo.add %352, %363 : tensor<2x8xi32>
      %365 = stablehlo.select %362, %364, %352 : tensor<2x8xi1>, tensor<2x8xi32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %367 = stablehlo.concatenate %360, %366, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %369 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x1024x192xf32>) -> tensor<256x1024x192x1x1xf32>
      %370 = "stablehlo.scatter"(%arg245, %368, %369) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %371 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %371 : tensor<f32>
      }) : (tensor<1024x192x16x16xf32>, tensor<256x2xi32>, tensor<256x1024x192x1x1xf32>) -> tensor<1024x192x16x16xf32>
      mhlo.return %370 : tensor<1024x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x192x16x16xf32>, tensor<2x8x2x8x1024x192xf32>) -> tensor<1024x192x16x16xf32>
    %255 = stablehlo.custom_call @__cudnn$convForward(%254, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x192x16x16xf32>, tensor<384x192x1x1xf32>) -> tuple<tensor<1024x384x16x16xf32>, tensor<0xui8>>
    %256 = stablehlo.get_tuple_element %255[0] : (tuple<tensor<1024x384x16x16xf32>, tensor<0xui8>>) -> tensor<1024x384x16x16xf32>
    %257 = "mhlo.fusion"(%6, %arg137, %arg136, %256, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<1024x384x16x16xf32>, %arg249: tensor<384xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x384x16x16xf32>
      %349 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x384x16x16xf32>
      %351 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x384x16x16xf32>
      %353 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<1024x384x16x16xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x384x16x16xf32>
      mhlo.return %354 : tensor<1024x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1024x384x16x16xf32>, tensor<384xf32>) -> tensor<1024x384x16x16xf32>
    %258 = stablehlo.custom_call @__cudnn$convForward(%257, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,768,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x384x16x16xf32>, tensor<768x384x1x1xf32>) -> tuple<tensor<1024x768x16x16xf32>, tensor<0xui8>>
    %259 = stablehlo.get_tuple_element %258[0] : (tuple<tensor<1024x768x16x16xf32>, tensor<0xui8>>) -> tensor<1024x768x16x16xf32>
    %260 = "mhlo.fusion"(%arg140, %arg139, %5, %259, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<1024x768x16x16xf32>, %arg249: tensor<768xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x768x16x16xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x768x16x16xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x768x16x16xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x16x16xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x768x16x16xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x768x16x16xf32>
      %356 = stablehlo.negate %354 : tensor<1024x768x16x16xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x768x16x16xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x768x16x16xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x768x16x16xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x768x16x16xf32>
      mhlo.return %360 : tensor<1024x768x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x768x16x16xf32>, tensor<768xf32>) -> tensor<1024x768x16x16xf32>
    %261 = stablehlo.custom_call @__cudnn$convForward(%260, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,768,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x768x16x16xf32>, tensor<768x1x3x3xf32>) -> tuple<tensor<1024x768x8x8xf32>, tensor<0xui8>>
    %262 = stablehlo.get_tuple_element %261[0] : (tuple<tensor<1024x768x8x8xf32>, tensor<0xui8>>) -> tensor<1024x768x8x8xf32>
    %263 = "mhlo.fusion"(%arg143, %arg142, %4, %262, %arg236) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<1024x768x8x8xf32>, %arg249: tensor<768xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x768x8x8xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x768x8x8xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x768x8x8xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<1024x768x8x8xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x768x8x8xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x768x8x8xf32>
      %356 = stablehlo.negate %354 : tensor<1024x768x8x8xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x768x8x8xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x768x8x8xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x768x8x8xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x768x8x8xf32>
      mhlo.return %360 : tensor<1024x768x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<1024x768x8x8xf32>, tensor<768xf32>) -> tensor<1024x768x8x8xf32>
    %264 = stablehlo.custom_call @__cudnn$convForward(%263, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x768x8x8xf32>, tensor<512x768x1x1xf32>) -> tuple<tensor<1024x512x8x8xf32>, tensor<0xui8>>
    %265 = stablehlo.get_tuple_element %264[0] : (tuple<tensor<1024x512x8x8xf32>, tensor<0xui8>>) -> tensor<1024x512x8x8xf32>
    %266 = "mhlo.fusion"(%3, %arg146, %arg145, %265, %arg238) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1024x512x8x8xf32>, %arg249: tensor<512xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x512x8x8xf32>
      %349 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x512x8x8xf32>
      %351 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x512x8x8xf32>
      %353 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x512x8x8xf32>
      mhlo.return %354 : tensor<1024x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1024x512x8x8xf32>, tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %267 = stablehlo.custom_call @__cudnn$convForward(%266, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x8x8xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<1024x512x8x8xf32>, tensor<0xui8>>
    %268 = stablehlo.get_tuple_element %267[0] : (tuple<tensor<1024x512x8x8xf32>, tensor<0xui8>>) -> tensor<1024x512x8x8xf32>
    %269 = "mhlo.fusion"(%arg149, %arg148, %2, %268, %arg240) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1024x512x8x8xf32>, %arg249: tensor<512xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x512x8x8xf32>
      %349 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x512x8x8xf32>
      %351 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x512x8x8xf32>
      %353 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x512x8x8xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %355 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x8x8xf32>
      %356 = stablehlo.negate %354 : tensor<1024x512x8x8xf32>
      %357 = stablehlo.exponential %356 : tensor<1024x512x8x8xf32>
      %358 = stablehlo.add %355, %357 : tensor<1024x512x8x8xf32>
      %359 = stablehlo.divide %355, %358 : tensor<1024x512x8x8xf32>
      %360 = stablehlo.multiply %354, %359 : tensor<1024x512x8x8xf32>
      mhlo.return %360 : tensor<1024x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1024x512x8x8xf32>, tensor<512xf32>) -> tensor<1024x512x8x8xf32>
    %270 = stablehlo.custom_call @__cudnn$convForward(%269, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x8x8xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<1024x256x8x8xf32>, tensor<0xui8>>
    %271 = stablehlo.get_tuple_element %270[0] : (tuple<tensor<1024x256x8x8xf32>, tensor<0xui8>>) -> tensor<1024x256x8x8xf32>
    %272 = "mhlo.fusion"(%271) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x256x8x8xf32>):
      %347 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %349 = stablehlo.multiply %347, %348 : tensor<4xi32>
      %350 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %351 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %352 = stablehlo.add %350, %351 : tensor<2x4xi32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %354 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %355 = stablehlo.compare LT, %353, %354 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %356 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %357 = stablehlo.add %353, %356 : tensor<1x1x2x4xi32>
      %358 = stablehlo.select %355, %357, %353 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %361 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %362 = stablehlo.compare LT, %352, %361 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %363 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %364 = stablehlo.add %352, %363 : tensor<2x4xi32>
      %365 = stablehlo.select %362, %364, %352 : tensor<2x4xi1>, tensor<2x4xi32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %367 = stablehlo.concatenate %360, %366, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %369 = "stablehlo.gather"(%arg245, %368) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 256, 1, 1>}> : (tensor<1024x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1024x256x1x1xf32>
      %370 = mhlo.bitcast %369 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1024x256x1x1xf32>) -> tensor<2x4x2x4x1024x256xf32>
      %371 = stablehlo.transpose %370, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x1024x256xf32>) -> tensor<1024x256x2x2x4x4xf32>
      mhlo.return %371 : tensor<1024x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x8x8xf32>) -> tensor<1024x256x2x2x4x4xf32>
    %273 = "mhlo.fusion"(%271) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x8x8xf32>):
      %347 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %349 = stablehlo.multiply %347, %348 : tensor<4xi32>
      %350 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %351 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %352 = stablehlo.add %350, %351 : tensor<2x4xi32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %354 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %355 = stablehlo.compare LT, %353, %354 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %356 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %357 = stablehlo.add %353, %356 : tensor<1x1x2x4xi32>
      %358 = stablehlo.select %355, %357, %353 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %361 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %362 = stablehlo.compare LT, %352, %361 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %363 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %364 = stablehlo.add %352, %363 : tensor<2x4xi32>
      %365 = stablehlo.select %362, %364, %352 : tensor<2x4xi1>, tensor<2x4xi32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %367 = stablehlo.concatenate %360, %366, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %369 = "stablehlo.gather"(%arg245, %368) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1024, 256, 1, 1>}> : (tensor<1024x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x1024x256x1x1xf32>
      %370 = mhlo.bitcast %369 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x1024x256x1x1xf32>) -> tensor<64x1024x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %371 = stablehlo.reduce(%370 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1024x256xf32>, tensor<f32>) -> tensor<64x1024xf32>
      mhlo.return %371 : tensor<64x1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x8x8xf32>) -> tensor<64x1024xf32>
    %274 = "mhlo.fusion"(%273) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x1024xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %347 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x1024xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %347 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<64x1024xf32>) -> tensor<1024xf32>
    %275 = "mhlo.fusion"(%274) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %276 = "mhlo.fusion"(%272, %275) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x2x2x4x4xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x2x2x4x4xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x2x2x4x4xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x2x2x4x4xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x256x2x2x4x4xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x2x2x4x4xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %278 = "mhlo.fusion"(%277, %arg151, %arg152, %272, %275) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x2x2x4x4xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1x1024x16384xf32>
      %349 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x2x2x4x4xf32>
      %350 = stablehlo.subtract %arg248, %349 : tensor<1024x256x2x2x4x4xf32>
      %351 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x2x2x4x4xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x256x2x2x4x4xf32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x256x2x2x4x4xf32>) -> tensor<1x1024x16384xf32>
      %354 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1x1024x16384xf32>
      %356 = stablehlo.multiply %353, %355 : tensor<1x1024x16384xf32>
      %357 = stablehlo.add %348, %356 : tensor<1x1024x16384xf32>
      mhlo.return %357 : tensor<1x1024x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x2x2x4x4xf32>, tensor<1024xf32>) -> tensor<1x1024x16384xf32>
    %279 = mhlo.bitcast %278 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x1024x16384xf32>) -> tensor<1024x256x4x16xf32>
    %280 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%279, %arg153, %arg154) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<1024x513x4x16xf32>, tensor<0xui8>>
    %281 = stablehlo.get_tuple_element %280[0] : (tuple<tensor<1024x513x4x16xf32>, tensor<0xui8>>) -> tensor<1024x513x4x16xf32>
    %282 = "mhlo.fusion"(%281) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x513x4x16xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 257:513, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:257, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x1x4x16xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x16xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x16xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x16xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x256x4x16xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x256x4x16xf32>, tensor<f32>) -> tensor<1024x256x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x256x4xf32>) -> tensor<1024x256x4x16xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x256x4x16xf32>
      mhlo.return %366 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %283 = mhlo.bitcast %272 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<1024x256x2x2x4x4xf32>) -> tensor<1024x256x4x16xf32>
    %284 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%282, %arg155, %arg156, %283) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>) -> tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>
    %285 = stablehlo.get_tuple_element %284[0] : (tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>) -> tensor<1024x256x4x16xf32>
    %286 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>) -> tensor<1024xf32>
    %287 = "mhlo.fusion"(%286) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %288 = "mhlo.fusion"(%285, %287) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x4x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %289 = "mhlo.fusion"(%288) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %290 = "mhlo.fusion"(%289, %arg157, %arg158, %285, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x4x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x256x4x16xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x256x4x16xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x256x4x16xf32>
      mhlo.return %354 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %291 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%290, %arg159, %arg160) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<1024x512x4x16xf32>, tensor<0xui8>>
    %292 = stablehlo.get_tuple_element %291[0] : (tuple<tensor<1024x512x4x16xf32>, tensor<0xui8>>) -> tensor<1024x512x4x16xf32>
    %293 = "mhlo.fusion"(%292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x4x16xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x512x4x16xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x512x4x16xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x512x4x16xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x512x4x16xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x512x4x16xf32>
      mhlo.return %352 : tensor<1024x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x512x4x16xf32>) -> tensor<1024x512x4x16xf32>
    %294 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%293, %arg161, %arg162, %285) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>) -> tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>
    %295 = stablehlo.get_tuple_element %294[0] : (tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>) -> tensor<1024x256x4x16xf32>
    %296 = "mhlo.fusion"(%295) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>) -> tensor<1024xf32>
    %297 = "mhlo.fusion"(%296) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %298 = "mhlo.fusion"(%295, %297) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x4x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %299 = "mhlo.fusion"(%298) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %300 = "mhlo.fusion"(%299, %arg163, %arg164, %295, %297) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x4x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x256x4x16xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x256x4x16xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x256x4x16xf32>
      mhlo.return %354 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %301 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%300, %arg165, %arg166) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<1024x513x4x16xf32>, tensor<0xui8>>
    %302 = stablehlo.get_tuple_element %301[0] : (tuple<tensor<1024x513x4x16xf32>, tensor<0xui8>>) -> tensor<1024x513x4x16xf32>
    %303 = "mhlo.fusion"(%302) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x513x4x16xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 257:513, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:257, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x1x4x16xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x16xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x16xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x16xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x256x4x16xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x256x4x16xf32>, tensor<f32>) -> tensor<1024x256x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x256x4xf32>) -> tensor<1024x256x4x16xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x256x4x16xf32>
      mhlo.return %366 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %304 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%303, %arg167, %arg168, %295) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>) -> tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>
    %305 = stablehlo.get_tuple_element %304[0] : (tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>) -> tensor<1024x256x4x16xf32>
    %306 = "mhlo.fusion"(%305) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>) -> tensor<1024xf32>
    %307 = "mhlo.fusion"(%306) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %308 = "mhlo.fusion"(%305, %307) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x4x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %309 = "mhlo.fusion"(%308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %310 = "mhlo.fusion"(%309, %arg169, %arg170, %305, %307) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x4x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x256x4x16xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x256x4x16xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x256x4x16xf32>
      mhlo.return %354 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %311 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%310, %arg171, %arg172) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<1024x512x4x16xf32>, tensor<0xui8>>
    %312 = stablehlo.get_tuple_element %311[0] : (tuple<tensor<1024x512x4x16xf32>, tensor<0xui8>>) -> tensor<1024x512x4x16xf32>
    %313 = "mhlo.fusion"(%312) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x4x16xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x512x4x16xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x512x4x16xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x512x4x16xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x512x4x16xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x512x4x16xf32>
      mhlo.return %352 : tensor<1024x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x512x4x16xf32>) -> tensor<1024x512x4x16xf32>
    %314 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%313, %arg173, %arg174, %305) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>) -> tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>
    %315 = stablehlo.get_tuple_element %314[0] : (tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>) -> tensor<1024x256x4x16xf32>
    %316 = "mhlo.fusion"(%315) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>) -> tensor<1024xf32>
    %317 = "mhlo.fusion"(%316) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %318 = "mhlo.fusion"(%315, %317) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x4x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %319 = "mhlo.fusion"(%318) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %320 = "mhlo.fusion"(%319, %arg175, %arg176, %315, %317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x4x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x256x4x16xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x256x4x16xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x256x4x16xf32>
      mhlo.return %354 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %321 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%320, %arg177, %arg178) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<1024x513x4x16xf32>, tensor<0xui8>>
    %322 = stablehlo.get_tuple_element %321[0] : (tuple<tensor<1024x513x4x16xf32>, tensor<0xui8>>) -> tensor<1024x513x4x16xf32>
    %323 = "mhlo.fusion"(%322) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x513x4x16xf32>):
      %347 = stablehlo.slice %arg245 [0:1024, 257:513, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.maximum %347, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.slice %arg245 [0:1024, 1:257, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.slice %arg245 [0:1024, 0:1, 0:4, 0:16] : (tensor<1024x513x4x16xf32>) -> tensor<1024x1x4x16xf32>
      %352 = mhlo.bitcast %351 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %353 = stablehlo.reduce(%352 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
      %355 = stablehlo.subtract %351, %354 : tensor<1024x1x4x16xf32>
      %356 = stablehlo.exponential %355 : tensor<1024x1x4x16xf32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %358 = stablehlo.reduce(%357 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x4x16xf32>, tensor<f32>) -> tensor<1024x4xf32>
      %359 = stablehlo.broadcast_in_dim %358, dims = [0, 2] : (tensor<1024x4xf32>) -> tensor<1024x1x4x16xf32>
      %360 = stablehlo.divide %356, %359 : tensor<1024x1x4x16xf32>
      %361 = mhlo.bitcast %360 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x1x4x16xf32>) -> tensor<1024x4x16xf32>
      %362 = stablehlo.broadcast_in_dim %361, dims = [0, 2, 3] : (tensor<1024x4x16xf32>) -> tensor<1024x256x4x16xf32>
      %363 = stablehlo.multiply %350, %362 : tensor<1024x256x4x16xf32>
      %364 = stablehlo.reduce(%363 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1024x256x4x16xf32>, tensor<f32>) -> tensor<1024x256x4xf32>
      %365 = stablehlo.broadcast_in_dim %364, dims = [0, 1, 2] : (tensor<1024x256x4xf32>) -> tensor<1024x256x4x16xf32>
      %366 = stablehlo.multiply %349, %365 : tensor<1024x256x4x16xf32>
      mhlo.return %366 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x513x4x16xf32>) -> tensor<1024x256x4x16xf32>
    %324 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%323, %arg179, %arg180, %315) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>) -> tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>
    %325 = stablehlo.get_tuple_element %324[0] : (tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>) -> tensor<1024x256x4x16xf32>
    %326 = "mhlo.fusion"(%325) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>) -> tensor<1024xf32>
    %327 = "mhlo.fusion"(%326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %328 = "mhlo.fusion"(%325, %327) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x4x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %329 = "mhlo.fusion"(%328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %330 = "mhlo.fusion"(%329, %arg181, %arg182, %325, %327) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x4x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x256x4x16xf32>
      %352 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x256x4x16xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x256x4x16xf32>
      mhlo.return %354 : tensor<1024x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
    %331 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%330, %arg183, %arg184) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<1024x512x4x16xf32>, tensor<0xui8>>
    %332 = stablehlo.get_tuple_element %331[0] : (tuple<tensor<1024x512x4x16xf32>, tensor<0xui8>>) -> tensor<1024x512x4x16xf32>
    %333 = "mhlo.fusion"(%332) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512x4x16xf32>
      %348 = stablehlo.negate %arg245 : tensor<1024x512x4x16xf32>
      %349 = stablehlo.exponential %348 : tensor<1024x512x4x16xf32>
      %350 = stablehlo.add %347, %349 : tensor<1024x512x4x16xf32>
      %351 = stablehlo.divide %347, %350 : tensor<1024x512x4x16xf32>
      %352 = stablehlo.multiply %arg245, %351 : tensor<1024x512x4x16xf32>
      mhlo.return %352 : tensor<1024x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x512x4x16xf32>) -> tensor<1024x512x4x16xf32>
    %334 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%333, %arg185, %arg186, %325) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>) -> tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>
    %335 = stablehlo.get_tuple_element %334[0] : (tuple<tensor<1024x256x4x16xf32>, tensor<0xui8>>) -> tensor<1024x256x4x16xf32>
    %336 = "mhlo.fusion"(%335) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>):
      %347 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>) -> tensor<1024xf32>
    %337 = "mhlo.fusion"(%336) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      mhlo.return %348 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %338 = "mhlo.fusion"(%335, %337) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x4x16xf32>, %arg246: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.subtract %arg245, %347 : tensor<1024x256x4x16xf32>
      %349 = stablehlo.multiply %348, %348 : tensor<1024x256x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %351 = stablehlo.reduce(%350 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x16384xf32>, tensor<f32>) -> tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %339 = "mhlo.fusion"(%338) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %349 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
      %350 = stablehlo.add %348, %349 : tensor<1024xf32>
      %351 = stablehlo.rsqrt %350 : tensor<1024xf32>
      mhlo.return %351 : tensor<1024xf32>
    }) {output_operand_aliasing = []} : (tensor<1024xf32>) -> tensor<1024xf32>
    %340 = "mhlo.fusion"(%arg187, %339, %arg188, %335, %337) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<1024xf32>, %arg247: tensor<256xf32>, %arg248: tensor<1024x256x4x16xf32>, %arg249: tensor<1024xf32>):
      %347 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %348 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %349 = stablehlo.subtract %arg248, %348 : tensor<1024x256x4x16xf32>
      %350 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<1024xf32>) -> tensor<1024x256x4x16xf32>
      %351 = stablehlo.multiply %349, %350 : tensor<1024x256x4x16xf32>
      %352 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<1024x256x4x16xf32>
      %353 = stablehlo.multiply %351, %352 : tensor<1024x256x4x16xf32>
      %354 = stablehlo.add %347, %353 : tensor<1024x256x4x16xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x256x4x16xf32>) -> tensor<1024x256x2x2x4x4xf32>
      %356 = stablehlo.transpose %355, dims = [2, 4, 3, 5, 0, 1] : (tensor<1024x256x2x2x4x4xf32>) -> tensor<2x4x2x4x1024x256xf32>
      mhlo.return %356 : tensor<2x4x2x4x1024x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<1024x256x4x16xf32>, tensor<1024xf32>) -> tensor<2x4x2x4x1024x256xf32>
    %341 = "mhlo.fusion"(%1, %340) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x256x8x8xf32>, %arg246: tensor<2x4x2x4x1024x256xf32>):
      %347 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %348 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %349 = stablehlo.multiply %347, %348 : tensor<4xi32>
      %350 = stablehlo.broadcast_in_dim %349, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %351 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %352 = stablehlo.add %350, %351 : tensor<2x4xi32>
      %353 = mhlo.bitcast %352 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %354 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %355 = stablehlo.compare LT, %353, %354 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %356 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %357 = stablehlo.add %353, %356 : tensor<1x1x2x4xi32>
      %358 = stablehlo.select %355, %357, %353 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %360 = stablehlo.broadcast_in_dim %359, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %361 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %362 = stablehlo.compare LT, %352, %361 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %363 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %364 = stablehlo.add %352, %363 : tensor<2x4xi32>
      %365 = stablehlo.select %362, %364, %352 : tensor<2x4xi1>, tensor<2x4xi32>
      %366 = stablehlo.broadcast_in_dim %365, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %367 = stablehlo.concatenate %360, %366, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %368 = mhlo.bitcast %367 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %369 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x1024x256xf32>) -> tensor<64x1024x256x1x1xf32>
      %370 = "stablehlo.scatter"(%arg245, %368, %369) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %371 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %371 : tensor<f32>
      }) : (tensor<1024x256x8x8xf32>, tensor<64x2xi32>, tensor<64x1024x256x1x1xf32>) -> tensor<1024x256x8x8xf32>
      mhlo.return %370 : tensor<1024x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x256x8x8xf32>, tensor<2x4x2x4x1024x256xf32>) -> tensor<1024x256x8x8xf32>
    %342 = stablehlo.custom_call @__cudnn$convForward(%341, %arg189) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[1024,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<1024x256x8x8xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<1024x512x8x8xf32>, tensor<0xui8>>
    %343 = stablehlo.get_tuple_element %342[0] : (tuple<tensor<1024x512x8x8xf32>, tensor<0xui8>>) -> tensor<1024x512x8x8xf32>
    %344 = "mhlo.fusion"(%0, %arg191, %arg190, %343, %arg242) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<1024x512x8x8xf32>, %arg249: tensor<512xf32>):
      %347 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %348 = stablehlo.subtract %arg248, %347 : tensor<1024x512x8x8xf32>
      %349 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %350 = stablehlo.multiply %348, %349 : tensor<1024x512x8x8xf32>
      %351 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %352 = stablehlo.multiply %350, %351 : tensor<1024x512x8x8xf32>
      %353 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<1024x512x8x8xf32>
      %354 = stablehlo.add %352, %353 : tensor<1024x512x8x8xf32>
      %355 = mhlo.bitcast %354 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1024x512x8x8xf32>) -> tensor<1024x512x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %356 = stablehlo.reduce(%355 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x512x64xf32>, tensor<f32>) -> tensor<1024x512xf32>
      mhlo.return %356 : tensor<1024x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1024x512x8x8xf32>, tensor<512xf32>) -> tensor<1024x512xf32>
    %345 = "mhlo.fusion"(%344, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<1024x512xf32>, %arg246: tensor<2x512xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024x512xf32>
      %348 = stablehlo.multiply %arg245, %347 : tensor<1024x512xf32>
      %349 = stablehlo.dot_general %348, %arg246, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<2x512xf32>) -> tensor<1024x2xf32>
      mhlo.return %349 : tensor<1024x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x512xf32>, tensor<2x512xf32>) -> tensor<1024x2xf32>
    %346 = "mhlo.fusion"(%345, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<1024x2xf32>, %arg246: tensor<2xf32>):
      %347 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<2xf32>) -> tensor<1024x2xf32>
      %348 = stablehlo.add %347, %arg245 : tensor<1024x2xf32>
      mhlo.return %348 : tensor<1024x2xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x2xf32>, tensor<2xf32>) -> tensor<1024x2xf32>
    return %346 : tensor<1024x2xf32>
  }
}
