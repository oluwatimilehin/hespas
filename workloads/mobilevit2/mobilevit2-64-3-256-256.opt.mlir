module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<64x512xf32>, %arg1: tensor<2x512xf32>) -> tensor<64x2xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64x512xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<2x512xf32>) -> tensor<64x2xf32>
    return %2 : tensor<64x2xf32>
  }
  func.func private @region_1.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<64x192x4x64xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x4x64xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x4x64xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<64x192x4x64xf32>) -> tensor<64x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<64x192x2x2x8x8xf32>, %arg1: tensor<64xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x192x2x2x8x8xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x192x2x2x8x8xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x192x2x2x8x8xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x192x2x2x8x8xf32>) -> tensor<64x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
    return %4 : tensor<64x128xf32>
  }
  func.func private @region_1.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<64x128x4x256xf32>, %arg1: tensor<64xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<64x128x4x256xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<64x128x4x256xf32>, %arg1: tensor<64xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<64x128x4x256xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<64x128x4x256xf32>, %arg1: tensor<64xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<64x128x4x256xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<64x128x4x256xf32>, %arg1: tensor<64xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<64x128x4x256xf32>) -> tensor<64x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %1 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<64x128x2x2x16x16xf32>, %arg1: tensor<64xf32>) -> tensor<64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x128x2x2x16x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x128x2x2x16x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x128x2x2x16x16xf32>) -> tensor<64x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
    return %4 : tensor<64x256xf32>
  }
  func.func private @region_1.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x8x8xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x512x8x8xf32>) -> tensor<64x512x64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x512x64xf32>, tensor<f32>) -> tensor<64x512xf32>
    return %12 : tensor<64x512xf32>
  }
  func.func private @region_1.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<64x256x4x16xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<64x256x4x16xf32>) -> tensor<64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<64x256x4x16xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<64x256x4x16xf32>) -> tensor<64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<64x256x4x16xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<64x256x4x16xf32>) -> tensor<64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<64x256x4x16xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<64x256x4x16xf32>) -> tensor<64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<64x256x4x16xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<64x256x4x16xf32>) -> tensor<64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<64x256x4x16xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<64x256x4x16xf32>) -> tensor<64xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<64x256x2x2x4x4xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<64xf32>) -> tensor<64x256x2x2x4x4xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x256x2x2x4x4xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<64x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x256x2x2x4x4xf32>) -> tensor<64x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @region_1.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<64x256x8x8xf32>) -> tensor<64x64xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x64x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x64x256x1x1xf32>) -> tensor<64x64x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x64x256xf32>, tensor<f32>) -> tensor<64x64xf32>
    return %24 : tensor<64x64xf32>
  }
  func.func private @region_1.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<64x192x16x16xf32>) -> tensor<256x64xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 192, 1, 1>}> : (tensor<64x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x64x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x64x192x1x1xf32>) -> tensor<256x64x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x64x192xf32>, tensor<f32>) -> tensor<256x64xf32>
    return %24 : tensor<256x64xf32>
  }
  func.func private @region_1.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<64x128x32x32xf32>) -> tensor<1024x64xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 128, 1, 1>}> : (tensor<64x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x64x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x64x128x1x1xf32>) -> tensor<1024x64x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x64x128xf32>, tensor<f32>) -> tensor<1024x64xf32>
    return %21 : tensor<1024x64xf32>
  }
  func.func private @region_1.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter(%arg0: tensor<64x256x8x8xf32>, %arg1: tensor<2x4x2x4x64x256xf32>) -> tensor<64x256x8x8xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x64x256xf32>) -> tensor<64x64x256x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<64x256x8x8xf32>, tensor<64x2xi32>, tensor<64x64x256x1x1xf32>) -> tensor<64x256x8x8xf32>
    return %23 : tensor<64x256x8x8xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x4x16xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<2x4x2x4x64x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6 = stablehlo.add %4, %5 : tensor<64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<64xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<64x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<64x256x4x16xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x256x2x2x4x4xf32>
    %14 = stablehlo.transpose %13, dims = [2, 4, 3, 5, 0, 1] : (tensor<64x256x2x2x4x4xf32>) -> tensor<2x4x2x4x64x256xf32>
    return %14 : tensor<2x4x2x4x64x256xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<64x256x8x8xf32>) -> tensor<64x256x2x2x4x4xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x64x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x64x256x1x1xf32>) -> tensor<2x4x2x4x64x256xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x64x256xf32>) -> tensor<64x256x2x2x4x4xf32>
    return %24 : tensor<64x256x2x2x4x4xf32>
  }
  func.func private @region_1.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.1(%arg0: tensor<64x192x16x16xf32>, %arg1: tensor<2x8x2x8x64x192xf32>) -> tensor<64x192x16x16xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x64x192xf32>) -> tensor<256x64x192x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<64x192x16x16xf32>, tensor<256x2xi32>, tensor<256x64x192x1x1xf32>) -> tensor<64x192x16x16xf32>
    return %23 : tensor<64x192x16x16xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<2x8x2x8x64x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x192x2x2x8x8xf32>
    %9 = stablehlo.transpose %8, dims = [2, 4, 3, 5, 0, 1] : (tensor<64x192x2x2x8x8xf32>) -> tensor<2x8x2x8x64x192xf32>
    return %9 : tensor<2x8x2x8x64x192xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<64x192x16x16xf32>) -> tensor<64x192x2x2x8x8xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 192, 1, 1>}> : (tensor<64x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x64x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x64x192x1x1xf32>) -> tensor<2x8x2x8x64x192xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x64x192xf32>) -> tensor<64x192x2x2x8x8xf32>
    return %24 : tensor<64x192x2x2x8x8xf32>
  }
  func.func private @region_1.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.2(%arg0: tensor<64x128x32x32xf32>, %arg1: tensor<2x16x2x16x64x128xf32>) -> tensor<64x128x32x32xf32> {
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
    %19 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x64x128xf32>) -> tensor<1024x64x128x1x1xf32>
    %20 = "stablehlo.scatter"(%arg0, %18, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %21 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }) : (tensor<64x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x64x128x1x1xf32>) -> tensor<64x128x32x32xf32>
    return %20 : tensor<64x128x32x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128x4x256xf32>, %arg4: tensor<64xf32>) -> tensor<2x16x2x16x64x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x128x4x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x128x2x2x16x16xf32>
    %9 = stablehlo.transpose %8, dims = [2, 4, 3, 5, 0, 1] : (tensor<64x128x2x2x16x16xf32>) -> tensor<2x16x2x16x64x128xf32>
    return %9 : tensor<2x16x2x16x64x128xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<64x128x32x32xf32>) -> tensor<64x128x2x2x16x16xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 128, 1, 1>}> : (tensor<64x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x64x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x64x128x1x1xf32>) -> tensor<2x16x2x16x64x128xf32>
    %21 = stablehlo.transpose %20, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x64x128xf32>) -> tensor<64x128x2x2x16x16xf32>
    return %21 : tensor<64x128x2x2x16x16xf32>
  }
  func.func private @fused_add(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128x2x2x16x16xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<1x64x131072xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x128x2x2x16x16xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x128x2x2x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x128x2x2x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x2x2x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x128x2x2x16x16xf32>) -> tensor<1x64x131072xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<1x64x131072xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x64x131072xf32>
    %10 = stablehlo.add %1, %9 : tensor<1x64x131072xf32>
    return %10 : tensor<1x64x131072xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x2x2x8x8xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<1x64x49152xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x2x2x8x8xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<64x192x2x2x8x8xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x2x2x8x8xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x192x2x2x8x8xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x192x2x2x8x8xf32>) -> tensor<1x64x49152xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<1x64x49152xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x64x49152xf32>
    %10 = stablehlo.add %1, %9 : tensor<1x64x49152xf32>
    return %10 : tensor<1x64x49152xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x2x2x4x4xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<1x64x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<1x64x16384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x2x2x4x4xf32>
    %3 = stablehlo.subtract %arg2, %2 : tensor<64x256x2x2x4x4xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %5 = stablehlo.multiply %arg4, %4 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64xf32>
    %8 = stablehlo.rsqrt %7 : tensor<64xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<64xf32>) -> tensor<64x256x2x2x4x4xf32>
    %10 = stablehlo.multiply %3, %9 : tensor<64x256x2x2x4x4xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x256x2x2x4x4xf32>) -> tensor<1x64x16384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<1x64x16384xf32>
    %14 = stablehlo.multiply %11, %13 : tensor<1x64x16384xf32>
    %15 = stablehlo.add %1, %14 : tensor<1x64x16384xf32>
    return %15 : tensor<1x64x16384xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x128x128xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<64x128x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<64x128x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x128x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x128x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x128x128x128xf32>
    return %13 : tensor<64x128x128x128xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<64x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<64x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x64x128x128xf32>
    return %13 : tensor<64x64x128x128xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<64x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<64x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x64x128x128xf32>
    return %13 : tensor<64x64x128x128xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<64x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<64x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x256x64x64xf32>
    return %13 : tensor<64x256x64x64xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<64x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<64x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x256x64x64xf32>
    return %13 : tensor<64x256x64x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<64x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<64x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x256x64x64xf32>
    return %13 : tensor<64x256x64x64xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<64x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x64x128x128xf32>
    return %7 : tensor<64x64x128x128xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x128x128xf32>, %arg4: tensor<32xf32>) -> tensor<64x32x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x32x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x32x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x32x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x32x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x32x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<64x32x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<64x32x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x32x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x32x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x32x128x128xf32>
    return %13 : tensor<64x32x128x128xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<64x128x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<64x128x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x128x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x128x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x128x64x64xf32>
    return %13 : tensor<64x128x64x64xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<64x512x32x32xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x512x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<64x512x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<64x512x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x512x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x512x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x512x32x32xf32>
    return %13 : tensor<64x512x32x32xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<64x128x64x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<64x128x64x64xf32>, %arg5: tensor<128xf32>) -> tensor<64x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<64x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x64x64xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<64x128x64x64xf32>
    return %8 : tensor<64x128x64x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<64x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<64x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x128x64x64xf32>
    return %7 : tensor<64x128x64x64xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<64x256x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<64x256x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x256x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x256x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x256x32x32xf32>
    return %13 : tensor<64x256x32x32xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<64x256x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<64x256x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x256x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x256x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x256x32x32xf32>
    return %13 : tensor<64x256x32x32xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<64x256x4x256xf32>) -> tensor<64x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<64x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x256x4x256xf32>
    return %5 : tensor<64x256x4x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<64x256x4x256xf32>) -> tensor<64x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<64x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x256x4x256xf32>
    return %5 : tensor<64x256x4x256xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x32x32xf32>
    return %7 : tensor<64x256x32x32xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<64x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<64x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x256x32x32xf32>
    return %7 : tensor<64x256x32x32xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<64x768x16x16xf32>, %arg4: tensor<768xf32>) -> tensor<64x768x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x768x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x768x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x768x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x768x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x768x16x16xf32>
    %9 = stablehlo.negate %7 : tensor<64x768x16x16xf32>
    %10 = stablehlo.exponential %9 : tensor<64x768x16x16xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x768x16x16xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x768x16x16xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x768x16x16xf32>
    return %13 : tensor<64x768x16x16xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<64x512x16x16xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x512x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x512x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x512x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x512x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x16x16xf32>
    %9 = stablehlo.negate %7 : tensor<64x512x16x16xf32>
    %10 = stablehlo.exponential %9 : tensor<64x512x16x16xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x512x16x16xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x512x16x16xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x512x16x16xf32>
    return %13 : tensor<64x512x16x16xf32>
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
  func.func private @fused_computation.222(%arg0: tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 129:257, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:129, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x256xf32>) -> tensor<64x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x128x4x256xf32>, tensor<f32>) -> tensor<64x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x128x4xf32>) -> tensor<64x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x128x4x256xf32>
    return %19 : tensor<64x128x4x256xf32>
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
  func.func private @fused_computation.225(%arg0: tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 129:257, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:129, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x256xf32>) -> tensor<64x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x128x4x256xf32>, tensor<f32>) -> tensor<64x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x128x4xf32>) -> tensor<64x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x128x4x256xf32>
    return %19 : tensor<64x128x4x256xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128x4x256xf32>, %arg4: tensor<64xf32>) -> tensor<64x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x128x4x256xf32>
    return %7 : tensor<64x128x4x256xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128x4x256xf32>, %arg4: tensor<64xf32>) -> tensor<64x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x128x4x256xf32>
    return %7 : tensor<64x128x4x256xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128x4x256xf32>, %arg4: tensor<64xf32>) -> tensor<64x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x128x4x256xf32>
    return %7 : tensor<64x128x4x256xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<64x384x16x16xf32>, %arg4: tensor<384xf32>) -> tensor<64x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x384x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x384x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x384x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x384x16x16xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x16x16xf32>
    %9 = stablehlo.negate %7 : tensor<64x384x16x16xf32>
    %10 = stablehlo.exponential %9 : tensor<64x384x16x16xf32>
    %11 = stablehlo.add %8, %10 : tensor<64x384x16x16xf32>
    %12 = stablehlo.divide %8, %11 : tensor<64x384x16x16xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<64x384x16x16xf32>
    return %13 : tensor<64x384x16x16xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<64x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x384x4x64xf32>
    return %5 : tensor<64x384x4x64xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<64x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x384x4x64xf32>
    return %5 : tensor<64x384x4x64xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<64x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x384x4x64xf32>
    return %5 : tensor<64x384x4x64xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<64x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x384x4x64xf32>
    return %5 : tensor<64x384x4x64xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<64x384x16x16xf32>, %arg4: tensor<384xf32>) -> tensor<64x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x384x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x384x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x384x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x384x16x16xf32>
    return %7 : tensor<64x384x16x16xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<64x384x16x16xf32>, %arg4: tensor<384xf32>) -> tensor<64x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<64x384x16x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<64x384x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<64x384x16x16xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %7 = stablehlo.add %5, %6 : tensor<64x384x16x16xf32>
    return %7 : tensor<64x384x16x16xf32>
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
  func.func private @fused_computation.257(%arg0: tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x192x4x64xf32>
    return %19 : tensor<64x192x4x64xf32>
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
  func.func private @fused_computation.258(%arg0: tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x192x4x64xf32>
    return %19 : tensor<64x192x4x64xf32>
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
  func.func private @fused_computation.259(%arg0: tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x192x4x64xf32>
    return %19 : tensor<64x192x4x64xf32>
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
  func.func private @fused_computation.260(%arg0: tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x192x4x64xf32>
    return %19 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<64xf32>, %arg1: tensor<192xf32>, %arg2: tensor<192xf32>, %arg3: tensor<64x192x4x64xf32>, %arg4: tensor<64xf32>) -> tensor<64x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<64x192x4x64xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<64x192x4x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<64x192x4x64xf32>
    %7 = stablehlo.add %0, %6 : tensor<64x192x4x64xf32>
    return %7 : tensor<64x192x4x64xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<64x768x8x8xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<64x768x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<64x768x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x768x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x768x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x768x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x768x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<64x768x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<64x768x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<64x768x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<64x768x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<64x768x8x8xf32>
    return %16 : tensor<64x768x8x8xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<64x512x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<64x512x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<64x512x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<64x512x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<64x512x8x8xf32>
    return %16 : tensor<64x512x8x8xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<64x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<64x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<64x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<64x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<64x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<64x512x8x8xf32>
    return %10 : tensor<64x512x8x8xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<64x512x4x16xf32>) -> tensor<64x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<64x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x512x4x16xf32>
    return %5 : tensor<64x512x4x16xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<64x512x4x16xf32>) -> tensor<64x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<64x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x512x4x16xf32>
    return %5 : tensor<64x512x4x16xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<64x512x4x16xf32>) -> tensor<64x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<64x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<64x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<64x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<64x512x4x16xf32>
    return %5 : tensor<64x512x4x16xf32>
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
  func.func private @fused_computation.319(%arg0: tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 257:513, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:257, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x16xf32>) -> tensor<64x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x256x4x16xf32>, tensor<f32>) -> tensor<64x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x256x4xf32>) -> tensor<64x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x256x4x16xf32>
    return %19 : tensor<64x256x4x16xf32>
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
  func.func private @fused_computation.320(%arg0: tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 257:513, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:257, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x16xf32>) -> tensor<64x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x256x4x16xf32>, tensor<f32>) -> tensor<64x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x256x4xf32>) -> tensor<64x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x256x4x16xf32>
    return %19 : tensor<64x256x4x16xf32>
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
  func.func private @fused_computation.321(%arg0: tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:64, 257:513, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<64x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:64, 1:257, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:64, 0:1, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<64x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<64x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<64x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<64x4x16xf32>) -> tensor<64x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<64x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x256x4x16xf32>, tensor<f32>) -> tensor<64x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<64x256x4xf32>) -> tensor<64x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<64x256x4x16xf32>
    return %19 : tensor<64x256x4x16xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x4x16xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6 = stablehlo.add %4, %5 : tensor<64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<64xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<64x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<64x256x4x16xf32>
    return %12 : tensor<64x256x4x16xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x4x16xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6 = stablehlo.add %4, %5 : tensor<64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<64xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<64x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<64x256x4x16xf32>
    return %12 : tensor<64x256x4x16xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x4x16xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6 = stablehlo.add %4, %5 : tensor<64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<64xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<64x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<64x256x4x16xf32>
    return %12 : tensor<64x256x4x16xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x4x16xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6 = stablehlo.add %4, %5 : tensor<64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<64xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<64x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<64x256x4x16xf32>
    return %12 : tensor<64x256x4x16xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<64x256x4x16xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tensor<64x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<64x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %6 = stablehlo.add %4, %5 : tensor<64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<64xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<64x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<64x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<64x256x4x16xf32>
    return %12 : tensor<64x256x4x16xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<768xf32>) -> tensor<768xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<768xf32>
    %2 = stablehlo.rsqrt %1 : tensor<768xf32>
    return %2 : tensor<768xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.3(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_rsqrt.4(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_rsqrt.5(%arg0: tensor<384xf32>) -> tensor<384xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<384xf32>
    %2 = stablehlo.rsqrt %1 : tensor<384xf32>
    return %2 : tensor<384xf32>
  }
  func.func private @fused_rsqrt.6(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.7(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.8(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.9(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.10(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.11(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.12(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_add.26(%arg0: tensor<64x2xf32>, %arg1: tensor<2xf32>) -> tensor<64x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<64x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<64x2xf32>
    return %1 : tensor<64x2xf32>
  }
  func.func private @fused_rsqrt.13(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.14(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.15(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.16(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.17(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.18(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.19(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.20(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.21(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.22(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.23(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.24(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.25(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.26(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.27(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.28(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.29(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.30(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.add %1, %2 : tensor<64xf32>
    %4 = stablehlo.rsqrt %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.36(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.37(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.38(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.39(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.40(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.41(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.42(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.43(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.44(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_multiply.45(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<64xf32>
    return %1 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.31(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.32(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.33(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.34(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_broadcast() -> tensor<64x192x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x16x16xf32>
    return %0 : tensor<64x192x16x16xf32>
  }
  func.func private @fused_broadcast.1() -> tensor<64x128x32x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x32x32xf32>
    return %0 : tensor<64x128x32x32xf32>
  }
  func.func private @fused_broadcast.2() -> tensor<64x256x8x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x8x8xf32>
    return %0 : tensor<64x256x8x8xf32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<1024x64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x64xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.52(%arg0: tensor<64x256xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.53(%arg0: tensor<256x64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x64xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.54(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.55(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.56(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.57(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.58(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.59(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.60(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.61(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.62(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.63(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.64(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.65(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.66(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.67(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.68(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.69(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.70(%arg0: tensor<64x128xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func private @fused_reduce.71(%arg0: tensor<64x64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x64xf32>, tensor<f32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<64x3x256x256xf32>) -> tensor<64x2xf32> {
    %0 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x8x8xf32>
      mhlo.return %336 : tensor<64x256x8x8xf32>
    }) {output_operand_aliasing = []} : () -> tensor<64x256x8x8xf32>
    %1 = "mhlo.fusion"(%arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<768xf32>
      %338 = stablehlo.rsqrt %337 : tensor<768xf32>
      mhlo.return %338 : tensor<768xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>) -> tensor<768xf32>
    %2 = "mhlo.fusion"(%arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<384xf32>
      %338 = stablehlo.rsqrt %337 : tensor<384xf32>
      mhlo.return %338 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %3 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x16x16xf32>
      mhlo.return %336 : tensor<64x192x16x16xf32>
    }) {output_operand_aliasing = []} : () -> tensor<64x192x16x16xf32>
    %4 = "mhlo.fusion"(%arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<384xf32>
      %338 = stablehlo.rsqrt %337 : tensor<384xf32>
      mhlo.return %338 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %5 = "mhlo.fusion"(%arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<384xf32>
      %338 = stablehlo.rsqrt %337 : tensor<384xf32>
      mhlo.return %338 : tensor<384xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>) -> tensor<384xf32>
    %6 = "mhlo.fusion"(%arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<512xf32>
      %338 = stablehlo.rsqrt %337 : tensor<512xf32>
      mhlo.return %338 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %7 = "mhlo.fusion"(%arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<512xf32>
      %338 = stablehlo.rsqrt %337 : tensor<512xf32>
      mhlo.return %338 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %8 = "mhlo.fusion"(%arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %9 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x32x32xf32>
      mhlo.return %336 : tensor<64x128x32x32xf32>
    }) {output_operand_aliasing = []} : () -> tensor<64x128x32x32xf32>
    %10 = "mhlo.fusion"(%arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %11 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %12 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %13 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %14 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<128xf32>
      %338 = stablehlo.rsqrt %337 : tensor<128xf32>
      mhlo.return %338 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %15 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<128xf32>
      %338 = stablehlo.rsqrt %337 : tensor<128xf32>
      mhlo.return %338 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %16 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<128xf32>
      %338 = stablehlo.rsqrt %337 : tensor<128xf32>
      mhlo.return %338 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %17 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<64xf32>
      %338 = stablehlo.rsqrt %337 : tensor<64xf32>
      mhlo.return %338 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %18 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<64xf32>
      %338 = stablehlo.rsqrt %337 : tensor<64xf32>
      mhlo.return %338 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %19 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<64xf32>
      %338 = stablehlo.rsqrt %337 : tensor<64xf32>
      mhlo.return %338 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %20 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<32xf32>
      %338 = stablehlo.rsqrt %337 : tensor<32xf32>
      mhlo.return %338 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %21 = stablehlo.custom_call @__cudnn$convForward(%arg244, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x3x256x256xf32>, tensor<32x3x3x3xf32>) -> tuple<tensor<64x32x128x128xf32>, tensor<0xui8>>
    %22 = stablehlo.get_tuple_element %21[0] : (tuple<tensor<64x32x128x128xf32>, tensor<0xui8>>) -> tensor<64x32x128x128xf32>
    %23 = "mhlo.fusion"(%arg2, %arg1, %20, %22, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<64x32x128x128xf32>, %arg249: tensor<32xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x32x128x128xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x32x128x128xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x32x128x128xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<32xf32>) -> tensor<64x32x128x128xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x32x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x32x128x128xf32>
      %345 = stablehlo.negate %343 : tensor<64x32x128x128xf32>
      %346 = stablehlo.exponential %345 : tensor<64x32x128x128xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x32x128x128xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x32x128x128xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x32x128x128xf32>
      mhlo.return %349 : tensor<64x32x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<64x32x128x128xf32>, tensor<32xf32>) -> tensor<64x32x128x128xf32>
    %24 = stablehlo.custom_call @__cudnn$convForward(%23, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x32x128x128xf32>, tensor<64x32x1x1xf32>) -> tuple<tensor<64x64x128x128xf32>, tensor<0xui8>>
    %25 = stablehlo.get_tuple_element %24[0] : (tuple<tensor<64x64x128x128xf32>, tensor<0xui8>>) -> tensor<64x64x128x128xf32>
    %26 = "mhlo.fusion"(%arg5, %arg4, %19, %25, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x64x128x128xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x64x128x128xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x64x128x128xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x64x128x128xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x128x128xf32>
      %345 = stablehlo.negate %343 : tensor<64x64x128x128xf32>
      %346 = stablehlo.exponential %345 : tensor<64x64x128x128xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x64x128x128xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x64x128x128xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x64x128x128xf32>
      mhlo.return %349 : tensor<64x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x128x128xf32>, tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %27 = stablehlo.custom_call @__cudnn$convForward(%26, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x128x128xf32>, tensor<64x1x3x3xf32>) -> tuple<tensor<64x64x128x128xf32>, tensor<0xui8>>
    %28 = stablehlo.get_tuple_element %27[0] : (tuple<tensor<64x64x128x128xf32>, tensor<0xui8>>) -> tensor<64x64x128x128xf32>
    %29 = "mhlo.fusion"(%arg8, %arg7, %18, %28, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x64x128x128xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x64x128x128xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x64x128x128xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x64x128x128xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64x128x128xf32>
      %345 = stablehlo.negate %343 : tensor<64x64x128x128xf32>
      %346 = stablehlo.exponential %345 : tensor<64x64x128x128xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x64x128x128xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x64x128x128xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x64x128x128xf32>
      mhlo.return %349 : tensor<64x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x128x128xf32>, tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %30 = stablehlo.custom_call @__cudnn$convForward(%29, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x128x128xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<64x64x128x128xf32>, tensor<0xui8>>
    %31 = stablehlo.get_tuple_element %30[0] : (tuple<tensor<64x64x128x128xf32>, tensor<0xui8>>) -> tensor<64x64x128x128xf32>
    %32 = "mhlo.fusion"(%17, %arg11, %arg10, %31, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x64x128x128xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x64x128x128xf32>
      %338 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x64x128x128xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x64x128x128xf32>
      %342 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<64x64x128x128xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x64x128x128xf32>
      mhlo.return %343 : tensor<64x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x128x128xf32>, tensor<64xf32>) -> tensor<64x64x128x128xf32>
    %33 = stablehlo.custom_call @__cudnn$convForward(%32, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<64x64x128x128xf32>, tensor<128x64x1x1xf32>) -> tuple<tensor<64x128x128x128xf32>, tensor<0xui8>>
    %34 = stablehlo.get_tuple_element %33[0] : (tuple<tensor<64x128x128x128xf32>, tensor<0xui8>>) -> tensor<64x128x128x128xf32>
    %35 = "mhlo.fusion"(%arg14, %arg13, %16, %34, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<64x128x128x128xf32>, %arg249: tensor<128xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x128x128x128xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x128x128x128xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x128x128x128xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x128x128xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x128x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x128x128xf32>
      %345 = stablehlo.negate %343 : tensor<64x128x128x128xf32>
      %346 = stablehlo.exponential %345 : tensor<64x128x128x128xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x128x128x128xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x128x128x128xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x128x128x128xf32>
      mhlo.return %349 : tensor<64x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x128x128xf32>, tensor<128xf32>) -> tensor<64x128x128x128xf32>
    %36 = stablehlo.custom_call @__cudnn$convForward(%35, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<64x128x64x64xf32>, tensor<0xui8>>
    %37 = stablehlo.get_tuple_element %36[0] : (tuple<tensor<64x128x64x64xf32>, tensor<0xui8>>) -> tensor<64x128x64x64xf32>
    %38 = "mhlo.fusion"(%arg17, %arg16, %15, %37, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<64x128x64x64xf32>, %arg249: tensor<128xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x128x64x64xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x128x64x64xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x128x64x64xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x128x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x64x64xf32>
      %345 = stablehlo.negate %343 : tensor<64x128x64x64xf32>
      %346 = stablehlo.exponential %345 : tensor<64x128x64x64xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x128x64x64xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x128x64x64xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x128x64x64xf32>
      mhlo.return %349 : tensor<64x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x64x64xf32>, tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %39 = stablehlo.custom_call @__cudnn$convForward(%38, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x64x64xf32>, tensor<128x128x1x1xf32>) -> tuple<tensor<64x128x64x64xf32>, tensor<0xui8>>
    %40 = stablehlo.get_tuple_element %39[0] : (tuple<tensor<64x128x64x64xf32>, tensor<0xui8>>) -> tensor<64x128x64x64xf32>
    %41 = "mhlo.fusion"(%14, %arg20, %arg19, %40, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<64x128x64x64xf32>, %arg249: tensor<128xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x128x64x64xf32>
      %338 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x128x64x64xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x128x64x64xf32>
      %342 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x128x64x64xf32>
      mhlo.return %343 : tensor<64x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x64x64xf32>, tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %42 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<128xf32>
      %338 = stablehlo.rsqrt %337 : tensor<128xf32>
      mhlo.return %338 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %43 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %44 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %337 = stablehlo.add %arg245, %336 : tensor<256xf32>
      %338 = stablehlo.rsqrt %337 : tensor<256xf32>
      mhlo.return %338 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %45 = stablehlo.custom_call @__cudnn$convForward(%41, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>
    %46 = stablehlo.get_tuple_element %45[0] : (tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>) -> tensor<64x256x64x64xf32>
    %47 = "mhlo.fusion"(%arg23, %arg22, %44, %46, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x64x64xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x64x64xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x64x64xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x64x64xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %345 = stablehlo.negate %343 : tensor<64x256x64x64xf32>
      %346 = stablehlo.exponential %345 : tensor<64x256x64x64xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x256x64x64xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x256x64x64xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x256x64x64xf32>
      mhlo.return %349 : tensor<64x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x64x64xf32>, tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %48 = stablehlo.custom_call @__cudnn$convForward(%47, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>
    %49 = stablehlo.get_tuple_element %48[0] : (tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>) -> tensor<64x256x64x64xf32>
    %50 = "mhlo.fusion"(%arg26, %arg25, %43, %49, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x64x64xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x64x64xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x64x64xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x64x64xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %345 = stablehlo.negate %343 : tensor<64x256x64x64xf32>
      %346 = stablehlo.exponential %345 : tensor<64x256x64x64xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x256x64x64xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x256x64x64xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x256x64x64xf32>
      mhlo.return %349 : tensor<64x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x64x64xf32>, tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %51 = stablehlo.custom_call @__cudnn$convForward(%50, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<64x128x64x64xf32>, tensor<0xui8>>
    %52 = stablehlo.get_tuple_element %51[0] : (tuple<tensor<64x128x64x64xf32>, tensor<0xui8>>) -> tensor<64x128x64x64xf32>
    %53 = "mhlo.fusion"(%41, %42, %arg29, %arg28, %52, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x128x64x64xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<128xf32>, %arg249: tensor<64x128x64x64xf32>, %arg250: tensor<128xf32>):
      %336 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %337 = stablehlo.subtract %arg249, %336 : tensor<64x128x64x64xf32>
      %338 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x128x64x64xf32>
      %340 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x128x64x64xf32>
      %342 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<64x128x64x64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x128x64x64xf32>
      %344 = stablehlo.add %arg245, %343 : tensor<64x128x64x64xf32>
      mhlo.return %344 : tensor<64x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<64x128x64x64xf32>, tensor<128xf32>) -> tensor<64x128x64x64xf32>
    %54 = stablehlo.custom_call @__cudnn$convForward(%53, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>
    %55 = stablehlo.get_tuple_element %54[0] : (tuple<tensor<64x256x64x64xf32>, tensor<0xui8>>) -> tensor<64x256x64x64xf32>
    %56 = "mhlo.fusion"(%arg32, %arg31, %13, %55, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x64x64xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x64x64xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x64x64xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x64x64xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x64x64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x64x64xf32>
      %345 = stablehlo.negate %343 : tensor<64x256x64x64xf32>
      %346 = stablehlo.exponential %345 : tensor<64x256x64x64xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x256x64x64xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x256x64x64xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x256x64x64xf32>
      mhlo.return %349 : tensor<64x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x64x64xf32>, tensor<256xf32>) -> tensor<64x256x64x64xf32>
    %57 = stablehlo.custom_call @__cudnn$convForward(%56, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>
    %58 = stablehlo.get_tuple_element %57[0] : (tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>) -> tensor<64x256x32x32xf32>
    %59 = "mhlo.fusion"(%arg35, %arg34, %12, %58, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x32x32xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x32x32xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x32x32xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x32x32xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x32x32xf32>
      %345 = stablehlo.negate %343 : tensor<64x256x32x32xf32>
      %346 = stablehlo.exponential %345 : tensor<64x256x32x32xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x256x32x32xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x256x32x32xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x256x32x32xf32>
      mhlo.return %349 : tensor<64x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x32x32xf32>, tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %60 = stablehlo.custom_call @__cudnn$convForward(%59, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x32x32xf32>, tensor<256x256x1x1xf32>) -> tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>
    %61 = stablehlo.get_tuple_element %60[0] : (tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>) -> tensor<64x256x32x32xf32>
    %62 = "mhlo.fusion"(%11, %arg38, %arg37, %61, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x32x32xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x32x32xf32>
      %338 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x32x32xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x32x32xf32>
      %342 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x32x32xf32>
      mhlo.return %343 : tensor<64x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x32x32xf32>, tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %63 = stablehlo.custom_call @__cudnn$convForward(%62, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x32x32xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>
    %64 = stablehlo.get_tuple_element %63[0] : (tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>) -> tensor<64x256x32x32xf32>
    %65 = "mhlo.fusion"(%arg41, %arg40, %10, %64, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x32x32xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x32x32xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x32x32xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x32x32xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x32x32xf32>
      %345 = stablehlo.negate %343 : tensor<64x256x32x32xf32>
      %346 = stablehlo.exponential %345 : tensor<64x256x32x32xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x256x32x32xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x256x32x32xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x256x32x32xf32>
      mhlo.return %349 : tensor<64x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x32x32xf32>, tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %66 = stablehlo.custom_call @__cudnn$convForward(%65, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x32x32xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<64x128x32x32xf32>, tensor<0xui8>>
    %67 = stablehlo.get_tuple_element %66[0] : (tuple<tensor<64x128x32x32xf32>, tensor<0xui8>>) -> tensor<64x128x32x32xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x32x32xf32>):
      %336 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %337 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %338 = stablehlo.add %336, %337 : tensor<2x16xi32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %340 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %341 = stablehlo.compare LT, %339, %340 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %342 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %343 = stablehlo.add %339, %342 : tensor<1x1x2x16xi32>
      %344 = stablehlo.select %341, %343, %339 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %345 = mhlo.bitcast %344 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %347 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %348 = stablehlo.compare LT, %338, %347 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %349 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %350 = stablehlo.add %338, %349 : tensor<2x16xi32>
      %351 = stablehlo.select %348, %350, %338 : tensor<2x16xi1>, tensor<2x16xi32>
      %352 = stablehlo.broadcast_in_dim %351, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %353 = stablehlo.concatenate %346, %352, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %354 = mhlo.bitcast %353 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %355 = "stablehlo.gather"(%arg245, %354) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 128, 1, 1>}> : (tensor<64x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x64x128x1x1xf32>
      %356 = mhlo.bitcast %355 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x64x128x1x1xf32>) -> tensor<2x16x2x16x64x128xf32>
      %357 = stablehlo.transpose %356, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x64x128xf32>) -> tensor<64x128x2x2x16x16xf32>
      mhlo.return %357 : tensor<64x128x2x2x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x32x32xf32>) -> tensor<64x128x2x2x16x16xf32>
    %69 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x32x32xf32>):
      %336 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %337 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %338 = stablehlo.add %336, %337 : tensor<2x16xi32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %340 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %341 = stablehlo.compare LT, %339, %340 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %342 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %343 = stablehlo.add %339, %342 : tensor<1x1x2x16xi32>
      %344 = stablehlo.select %341, %343, %339 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %345 = mhlo.bitcast %344 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %347 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %348 = stablehlo.compare LT, %338, %347 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %349 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %350 = stablehlo.add %338, %349 : tensor<2x16xi32>
      %351 = stablehlo.select %348, %350, %338 : tensor<2x16xi1>, tensor<2x16xi32>
      %352 = stablehlo.broadcast_in_dim %351, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %353 = stablehlo.concatenate %346, %352, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %354 = mhlo.bitcast %353 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %355 = "stablehlo.gather"(%arg245, %354) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 128, 1, 1>}> : (tensor<64x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x64x128x1x1xf32>
      %356 = mhlo.bitcast %355 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x64x128x1x1xf32>) -> tensor<1024x64x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %357 = stablehlo.reduce(%356 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x64x128xf32>, tensor<f32>) -> tensor<1024x64xf32>
      mhlo.return %357 : tensor<1024x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x32x32xf32>) -> tensor<1024x64xf32>
    %70 = "mhlo.fusion"(%69) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x64xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x64xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x64xf32>) -> tensor<64xf32>
    %71 = "mhlo.fusion"(%70) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %72 = "mhlo.fusion"(%68, %71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x2x2x16x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x128x2x2x16x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x128x2x2x16x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x128x2x2x16x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x128x2x2x16x16xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %340 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x2x2x16x16xf32>, tensor<64xf32>) -> tensor<64x256xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %74 = "mhlo.fusion"(%73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %75 = "mhlo.fusion"(%arg44, %arg43, %74, %68, %71) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x128x2x2x16x16xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %337 = mhlo.bitcast %336 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<1x64x131072xf32>
      %338 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x128x2x2x16x16xf32>
      %339 = stablehlo.subtract %arg248, %338 : tensor<64x128x2x2x16x16xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<64xf32>) -> tensor<64x128x2x2x16x16xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x128x2x2x16x16xf32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x128x2x2x16x16xf32>) -> tensor<1x64x131072xf32>
      %343 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %344 = mhlo.bitcast %343 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<1x64x131072xf32>
      %345 = stablehlo.multiply %342, %344 : tensor<1x64x131072xf32>
      %346 = stablehlo.add %337, %345 : tensor<1x64x131072xf32>
      mhlo.return %346 : tensor<1x64x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64x128x2x2x16x16xf32>, tensor<64xf32>) -> tensor<1x64x131072xf32>
    %76 = mhlo.bitcast %75 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x131072xf32>) -> tensor<64x128x4x256xf32>
    %77 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%76, %arg45, %arg46) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<64x257x4x256xf32>, tensor<0xui8>>
    %78 = stablehlo.get_tuple_element %77[0] : (tuple<tensor<64x257x4x256xf32>, tensor<0xui8>>) -> tensor<64x257x4x256xf32>
    %79 = "mhlo.fusion"(%78) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x257x4x256xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 129:257, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x4x256xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x128x4x256xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:129, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x1x4x256xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x256xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x256xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x256xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x256xf32>) -> tensor<64x128x4x256xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x128x4x256xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x128x4x256xf32>, tensor<f32>) -> tensor<64x128x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x128x4xf32>) -> tensor<64x128x4x256xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x128x4x256xf32>
      mhlo.return %355 : tensor<64x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
    %80 = mhlo.bitcast %68 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x128x2x2x16x16xf32>) -> tensor<64x128x4x256xf32>
    %81 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%79, %arg47, %arg48, %80) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<64x128x4x256xf32>) -> tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>
    %82 = stablehlo.get_tuple_element %81[0] : (tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>) -> tensor<64x128x4x256xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %337 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>) -> tensor<64x256xf32>
    %84 = "mhlo.fusion"(%83) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %85 = "mhlo.fusion"(%84) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %86 = "mhlo.fusion"(%82, %85) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x128x4x256xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x128x4x256xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %340 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x256xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %88 = "mhlo.fusion"(%87) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %89 = "mhlo.fusion"(%arg50, %arg49, %88, %82, %85) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x128x4x256xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x128x4x256xf32>
      %339 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x128x4x256xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x128x4x256xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x128x4x256xf32>
      mhlo.return %343 : tensor<64x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %90 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%89, %arg51, %arg52) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<64x256x4x256xf32>, tensor<0xui8>>
    %91 = stablehlo.get_tuple_element %90[0] : (tuple<tensor<64x256x4x256xf32>, tensor<0xui8>>) -> tensor<64x256x4x256xf32>
    %92 = "mhlo.fusion"(%91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x256xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x256x4x256xf32>
      %338 = stablehlo.exponential %337 : tensor<64x256x4x256xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x256x4x256xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x256x4x256xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x256x4x256xf32>
      mhlo.return %341 : tensor<64x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x256xf32>) -> tensor<64x256x4x256xf32>
    %93 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%92, %arg53, %arg54, %82) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<64x128x4x256xf32>) -> tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>
    %94 = stablehlo.get_tuple_element %93[0] : (tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>) -> tensor<64x128x4x256xf32>
    %95 = "mhlo.fusion"(%94) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %337 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>) -> tensor<64x256xf32>
    %96 = "mhlo.fusion"(%95) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %97 = "mhlo.fusion"(%96) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %98 = "mhlo.fusion"(%94, %97) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x128x4x256xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x128x4x256xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %340 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x256xf32>
    %99 = "mhlo.fusion"(%98) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %100 = "mhlo.fusion"(%99) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %101 = "mhlo.fusion"(%arg56, %arg55, %100, %94, %97) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x128x4x256xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x128x4x256xf32>
      %339 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x128x4x256xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x128x4x256xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x128x4x256xf32>
      mhlo.return %343 : tensor<64x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %102 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%101, %arg57, %arg58) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<64x257x4x256xf32>, tensor<0xui8>>
    %103 = stablehlo.get_tuple_element %102[0] : (tuple<tensor<64x257x4x256xf32>, tensor<0xui8>>) -> tensor<64x257x4x256xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x257x4x256xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 129:257, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x128x4x256xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x128x4x256xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:129, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:256] : (tensor<64x257x4x256xf32>) -> tensor<64x1x4x256xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x256xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x256xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x256xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x256xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x256xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x256xf32>) -> tensor<64x4x256xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x256xf32>) -> tensor<64x128x4x256xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x128x4x256xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x128x4x256xf32>, tensor<f32>) -> tensor<64x128x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x128x4xf32>) -> tensor<64x128x4x256xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x128x4x256xf32>
      mhlo.return %355 : tensor<64x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x257x4x256xf32>) -> tensor<64x128x4x256xf32>
    %105 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%104, %arg59, %arg60, %94) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<64x128x4x256xf32>) -> tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>
    %106 = stablehlo.get_tuple_element %105[0] : (tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>) -> tensor<64x128x4x256xf32>
    %107 = "mhlo.fusion"(%106) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %337 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>) -> tensor<64x256xf32>
    %108 = "mhlo.fusion"(%107) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %109 = "mhlo.fusion"(%108) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %110 = "mhlo.fusion"(%106, %109) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x128x4x256xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x128x4x256xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %340 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x256xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %112 = "mhlo.fusion"(%111) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %113 = "mhlo.fusion"(%arg62, %arg61, %112, %106, %109) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x128x4x256xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x128x4x256xf32>
      %339 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x128x4x256xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x128x4x256xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x128x4x256xf32>
      mhlo.return %343 : tensor<64x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x128x4x256xf32>
    %114 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%113, %arg63, %arg64) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<64x256x4x256xf32>, tensor<0xui8>>
    %115 = stablehlo.get_tuple_element %114[0] : (tuple<tensor<64x256x4x256xf32>, tensor<0xui8>>) -> tensor<64x256x4x256xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x256xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x256x4x256xf32>
      %338 = stablehlo.exponential %337 : tensor<64x256x4x256xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x256x4x256xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x256x4x256xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x256x4x256xf32>
      mhlo.return %341 : tensor<64x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x256xf32>) -> tensor<64x256x4x256xf32>
    %117 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%116, %arg65, %arg66, %106) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<64x128x4x256xf32>) -> tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>
    %118 = stablehlo.get_tuple_element %117[0] : (tuple<tensor<64x128x4x256xf32>, tensor<0xui8>>) -> tensor<64x128x4x256xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %337 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>) -> tensor<64x256xf32>
    %120 = "mhlo.fusion"(%119) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %121 = "mhlo.fusion"(%120) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %122 = "mhlo.fusion"(%118, %121) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x4x256xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x128x4x256xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x128x4x256xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x256x512xf32>, tensor<f32>) -> tensor<64x256xf32>
      mhlo.return %340 : tensor<64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<64x256xf32>
    %123 = "mhlo.fusion"(%122) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x256xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256xf32>) -> tensor<64xf32>
    %124 = "mhlo.fusion"(%123) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %125 = "mhlo.fusion"(%arg68, %arg67, %124, %118, %121) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<64xf32>, %arg248: tensor<64x128x4x256xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x128x4x256xf32>
      %339 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<64xf32>) -> tensor<64x128x4x256xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x128x4x256xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<64x128x4x256xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x128x4x256xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x128x4x256xf32>
      %344 = mhlo.bitcast %343 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x128x4x256xf32>) -> tensor<64x128x2x2x16x16xf32>
      %345 = stablehlo.transpose %344, dims = [2, 4, 3, 5, 0, 1] : (tensor<64x128x2x2x16x16xf32>) -> tensor<2x16x2x16x64x128xf32>
      mhlo.return %345 : tensor<2x16x2x16x64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<64xf32>, tensor<64x128x4x256xf32>, tensor<64xf32>) -> tensor<2x16x2x16x64x128xf32>
    %126 = "mhlo.fusion"(%9, %125) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128x32x32xf32>, %arg246: tensor<2x16x2x16x64x128xf32>):
      %336 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %337 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %338 = stablehlo.add %336, %337 : tensor<2x16xi32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %340 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %341 = stablehlo.compare LT, %339, %340 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %342 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %343 = stablehlo.add %339, %342 : tensor<1x1x2x16xi32>
      %344 = stablehlo.select %341, %343, %339 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %345 = mhlo.bitcast %344 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %347 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %348 = stablehlo.compare LT, %338, %347 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %349 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %350 = stablehlo.add %338, %349 : tensor<2x16xi32>
      %351 = stablehlo.select %348, %350, %338 : tensor<2x16xi1>, tensor<2x16xi32>
      %352 = stablehlo.broadcast_in_dim %351, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %353 = stablehlo.concatenate %346, %352, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %354 = mhlo.bitcast %353 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %355 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x64x128xf32>) -> tensor<1024x64x128x1x1xf32>
      %356 = "stablehlo.scatter"(%arg245, %354, %355) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %357 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %357 : tensor<f32>
      }) : (tensor<64x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x64x128x1x1xf32>) -> tensor<64x128x32x32xf32>
      mhlo.return %356 : tensor<64x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128x32x32xf32>, tensor<2x16x2x16x64x128xf32>) -> tensor<64x128x32x32xf32>
    %127 = stablehlo.custom_call @__cudnn$convForward(%126, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x128x32x32xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>
    %128 = stablehlo.get_tuple_element %127[0] : (tuple<tensor<64x256x32x32xf32>, tensor<0xui8>>) -> tensor<64x256x32x32xf32>
    %129 = "mhlo.fusion"(%8, %arg71, %arg70, %128, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<64x256x32x32xf32>, %arg249: tensor<256xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x256x32x32xf32>
      %338 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x256x32x32xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x256x32x32xf32>
      %342 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x32x32xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x256x32x32xf32>
      mhlo.return %343 : tensor<64x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x32x32xf32>, tensor<256xf32>) -> tensor<64x256x32x32xf32>
    %130 = stablehlo.custom_call @__cudnn$convForward(%129, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x32x32xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<64x512x32x32xf32>, tensor<0xui8>>
    %131 = stablehlo.get_tuple_element %130[0] : (tuple<tensor<64x512x32x32xf32>, tensor<0xui8>>) -> tensor<64x512x32x32xf32>
    %132 = "mhlo.fusion"(%arg74, %arg73, %7, %131, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<64x512x32x32xf32>, %arg249: tensor<512xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x512x32x32xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x512x32x32xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x512x32x32xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<64x512x32x32xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x512x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x32x32xf32>
      %345 = stablehlo.negate %343 : tensor<64x512x32x32xf32>
      %346 = stablehlo.exponential %345 : tensor<64x512x32x32xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x512x32x32xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x512x32x32xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x512x32x32xf32>
      mhlo.return %349 : tensor<64x512x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x32x32xf32>, tensor<512xf32>) -> tensor<64x512x32x32xf32>
    %133 = stablehlo.custom_call @__cudnn$convForward(%132, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x32x32xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<64x512x16x16xf32>, tensor<0xui8>>
    %134 = stablehlo.get_tuple_element %133[0] : (tuple<tensor<64x512x16x16xf32>, tensor<0xui8>>) -> tensor<64x512x16x16xf32>
    %135 = "mhlo.fusion"(%arg77, %arg76, %6, %134, %arg226) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<64x512x16x16xf32>, %arg249: tensor<512xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x512x16x16xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x512x16x16xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x512x16x16xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<64x512x16x16xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x512x16x16xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x16x16xf32>
      %345 = stablehlo.negate %343 : tensor<64x512x16x16xf32>
      %346 = stablehlo.exponential %345 : tensor<64x512x16x16xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x512x16x16xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x512x16x16xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x512x16x16xf32>
      mhlo.return %349 : tensor<64x512x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<64x512x16x16xf32>, tensor<512xf32>) -> tensor<64x512x16x16xf32>
    %136 = stablehlo.custom_call @__cudnn$convForward(%135, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x16x16xf32>, tensor<384x512x1x1xf32>) -> tuple<tensor<64x384x16x16xf32>, tensor<0xui8>>
    %137 = stablehlo.get_tuple_element %136[0] : (tuple<tensor<64x384x16x16xf32>, tensor<0xui8>>) -> tensor<64x384x16x16xf32>
    %138 = "mhlo.fusion"(%5, %arg80, %arg79, %137, %arg228) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<64x384x16x16xf32>, %arg249: tensor<384xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x384x16x16xf32>
      %338 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x384x16x16xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x384x16x16xf32>
      %342 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x384x16x16xf32>
      mhlo.return %343 : tensor<64x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x16x16xf32>, tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %139 = stablehlo.custom_call @__cudnn$convForward(%138, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x16x16xf32>, tensor<384x1x3x3xf32>) -> tuple<tensor<64x384x16x16xf32>, tensor<0xui8>>
    %140 = stablehlo.get_tuple_element %139[0] : (tuple<tensor<64x384x16x16xf32>, tensor<0xui8>>) -> tensor<64x384x16x16xf32>
    %141 = "mhlo.fusion"(%arg83, %arg82, %4, %140, %arg230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<64x384x16x16xf32>, %arg249: tensor<384xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x384x16x16xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x384x16x16xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x384x16x16xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x384x16x16xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x16x16xf32>
      %345 = stablehlo.negate %343 : tensor<64x384x16x16xf32>
      %346 = stablehlo.exponential %345 : tensor<64x384x16x16xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x384x16x16xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x384x16x16xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x384x16x16xf32>
      mhlo.return %349 : tensor<64x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x16x16xf32>, tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %142 = stablehlo.custom_call @__cudnn$convForward(%141, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x16x16xf32>, tensor<192x384x1x1xf32>) -> tuple<tensor<64x192x16x16xf32>, tensor<0xui8>>
    %143 = stablehlo.get_tuple_element %142[0] : (tuple<tensor<64x192x16x16xf32>, tensor<0xui8>>) -> tensor<64x192x16x16xf32>
    %144 = "mhlo.fusion"(%143) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x16x16xf32>):
      %336 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %337 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %338 = stablehlo.multiply %336, %337 : tensor<8xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %340 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %341 = stablehlo.add %339, %340 : tensor<2x8xi32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %343 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %344 = stablehlo.compare LT, %342, %343 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %345 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %346 = stablehlo.add %342, %345 : tensor<1x1x2x8xi32>
      %347 = stablehlo.select %344, %346, %342 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %350 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %351 = stablehlo.compare LT, %341, %350 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %352 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %353 = stablehlo.add %341, %352 : tensor<2x8xi32>
      %354 = stablehlo.select %351, %353, %341 : tensor<2x8xi1>, tensor<2x8xi32>
      %355 = stablehlo.broadcast_in_dim %354, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %356 = stablehlo.concatenate %349, %355, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %358 = "stablehlo.gather"(%arg245, %357) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 192, 1, 1>}> : (tensor<64x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x64x192x1x1xf32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x64x192x1x1xf32>) -> tensor<2x8x2x8x64x192xf32>
      %360 = stablehlo.transpose %359, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x64x192xf32>) -> tensor<64x192x2x2x8x8xf32>
      mhlo.return %360 : tensor<64x192x2x2x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x16x16xf32>) -> tensor<64x192x2x2x8x8xf32>
    %145 = "mhlo.fusion"(%143) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x16x16xf32>):
      %336 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %337 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %338 = stablehlo.multiply %336, %337 : tensor<8xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %340 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %341 = stablehlo.add %339, %340 : tensor<2x8xi32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %343 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %344 = stablehlo.compare LT, %342, %343 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %345 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %346 = stablehlo.add %342, %345 : tensor<1x1x2x8xi32>
      %347 = stablehlo.select %344, %346, %342 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %350 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %351 = stablehlo.compare LT, %341, %350 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %352 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %353 = stablehlo.add %341, %352 : tensor<2x8xi32>
      %354 = stablehlo.select %351, %353, %341 : tensor<2x8xi1>, tensor<2x8xi32>
      %355 = stablehlo.broadcast_in_dim %354, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %356 = stablehlo.concatenate %349, %355, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %358 = "stablehlo.gather"(%arg245, %357) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 192, 1, 1>}> : (tensor<64x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x64x192x1x1xf32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x64x192x1x1xf32>) -> tensor<256x64x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %360 = stablehlo.reduce(%359 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x64x192xf32>, tensor<f32>) -> tensor<256x64xf32>
      mhlo.return %360 : tensor<256x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x16x16xf32>) -> tensor<256x64xf32>
    %146 = "mhlo.fusion"(%145) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256x64xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x64xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<256x64xf32>) -> tensor<64xf32>
    %147 = "mhlo.fusion"(%146) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %148 = "mhlo.fusion"(%144, %147) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x2x2x8x8xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x2x2x8x8xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x2x2x8x8xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x2x2x8x8xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x192x2x2x8x8xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x2x2x8x8xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %150 = "mhlo.fusion"(%149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %151 = "mhlo.fusion"(%150, %arg85, %arg86, %144, %147) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x2x2x8x8xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = mhlo.bitcast %336 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<1x64x49152xf32>
      %338 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x2x2x8x8xf32>
      %339 = stablehlo.subtract %arg248, %338 : tensor<64x192x2x2x8x8xf32>
      %340 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x2x2x8x8xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x192x2x2x8x8xf32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x192x2x2x8x8xf32>) -> tensor<1x64x49152xf32>
      %343 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %344 = mhlo.bitcast %343 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<1x64x49152xf32>
      %345 = stablehlo.multiply %342, %344 : tensor<1x64x49152xf32>
      %346 = stablehlo.add %337, %345 : tensor<1x64x49152xf32>
      mhlo.return %346 : tensor<1x64x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x2x2x8x8xf32>, tensor<64xf32>) -> tensor<1x64x49152xf32>
    %152 = mhlo.bitcast %151 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x49152xf32>) -> tensor<64x192x4x64xf32>
    %153 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%152, %arg87, %arg88) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>
    %154 = stablehlo.get_tuple_element %153[0] : (tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>) -> tensor<64x385x4x64xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x385x4x64xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x64xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x64xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x192x4x64xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x192x4x64xf32>
      mhlo.return %355 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %156 = mhlo.bitcast %144 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x192x2x2x8x8xf32>) -> tensor<64x192x4x64xf32>
    %157 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%155, %arg89, %arg90, %156) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %158 = stablehlo.get_tuple_element %157[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %160 = "mhlo.fusion"(%159) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %161 = "mhlo.fusion"(%160) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %162 = "mhlo.fusion"(%158, %161) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %164 = "mhlo.fusion"(%163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %165 = "mhlo.fusion"(%164, %arg91, %arg92, %158, %161) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %166 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%165, %arg93, %arg94) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>
    %167 = stablehlo.get_tuple_element %166[0] : (tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>) -> tensor<64x384x4x64xf32>
    %168 = "mhlo.fusion"(%167) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x384x4x64xf32>
      %338 = stablehlo.exponential %337 : tensor<64x384x4x64xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x384x4x64xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x384x4x64xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x384x4x64xf32>
      mhlo.return %341 : tensor<64x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32>
    %169 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%168, %arg95, %arg96, %158) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %170 = stablehlo.get_tuple_element %169[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %171 = "mhlo.fusion"(%170) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %172 = "mhlo.fusion"(%171) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %173 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %174 = "mhlo.fusion"(%170, %173) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %176 = "mhlo.fusion"(%175) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %177 = "mhlo.fusion"(%176, %arg97, %arg98, %170, %173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %178 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%177, %arg99, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>
    %179 = stablehlo.get_tuple_element %178[0] : (tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>) -> tensor<64x385x4x64xf32>
    %180 = "mhlo.fusion"(%179) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x385x4x64xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x64xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x64xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x192x4x64xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x192x4x64xf32>
      mhlo.return %355 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %181 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%180, %arg101, %arg102, %170) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %182 = stablehlo.get_tuple_element %181[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %184 = "mhlo.fusion"(%183) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %185 = "mhlo.fusion"(%184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %186 = "mhlo.fusion"(%182, %185) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %187 = "mhlo.fusion"(%186) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %188 = "mhlo.fusion"(%187) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %189 = "mhlo.fusion"(%188, %arg103, %arg104, %182, %185) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %190 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%189, %arg105, %arg106) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>
    %191 = stablehlo.get_tuple_element %190[0] : (tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>) -> tensor<64x384x4x64xf32>
    %192 = "mhlo.fusion"(%191) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x384x4x64xf32>
      %338 = stablehlo.exponential %337 : tensor<64x384x4x64xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x384x4x64xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x384x4x64xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x384x4x64xf32>
      mhlo.return %341 : tensor<64x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32>
    %193 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%192, %arg107, %arg108, %182) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %194 = stablehlo.get_tuple_element %193[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %196 = "mhlo.fusion"(%195) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %197 = "mhlo.fusion"(%196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %198 = "mhlo.fusion"(%194, %197) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %200 = "mhlo.fusion"(%199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %201 = "mhlo.fusion"(%200, %arg109, %arg110, %194, %197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %202 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%201, %arg111, %arg112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>
    %203 = stablehlo.get_tuple_element %202[0] : (tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>) -> tensor<64x385x4x64xf32>
    %204 = "mhlo.fusion"(%203) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x385x4x64xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x64xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x64xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x192x4x64xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x192x4x64xf32>
      mhlo.return %355 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %205 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%204, %arg113, %arg114, %194) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %206 = stablehlo.get_tuple_element %205[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %207 = "mhlo.fusion"(%206) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %209 = "mhlo.fusion"(%208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %210 = "mhlo.fusion"(%206, %209) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %212 = "mhlo.fusion"(%211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %213 = "mhlo.fusion"(%212, %arg115, %arg116, %206, %209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %214 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%213, %arg117, %arg118) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>
    %215 = stablehlo.get_tuple_element %214[0] : (tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>) -> tensor<64x384x4x64xf32>
    %216 = "mhlo.fusion"(%215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x384x4x64xf32>
      %338 = stablehlo.exponential %337 : tensor<64x384x4x64xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x384x4x64xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x384x4x64xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x384x4x64xf32>
      mhlo.return %341 : tensor<64x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32>
    %217 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%216, %arg119, %arg120, %206) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %218 = stablehlo.get_tuple_element %217[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %219 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %220 = "mhlo.fusion"(%219) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %221 = "mhlo.fusion"(%220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %222 = "mhlo.fusion"(%218, %221) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %223 = "mhlo.fusion"(%222) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %224 = "mhlo.fusion"(%223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %225 = "mhlo.fusion"(%224, %arg121, %arg122, %218, %221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %226 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%225, %arg123, %arg124) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>
    %227 = stablehlo.get_tuple_element %226[0] : (tuple<tensor<64x385x4x64xf32>, tensor<0xui8>>) -> tensor<64x385x4x64xf32>
    %228 = "mhlo.fusion"(%227) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x385x4x64xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 193:385, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:193, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:64] : (tensor<64x385x4x64xf32>) -> tensor<64x1x4x64xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x64xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x64xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x64xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x64xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x64xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x64xf32>) -> tensor<64x4x64xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x64xf32>) -> tensor<64x192x4x64xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x192x4x64xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x192x4x64xf32>, tensor<f32>) -> tensor<64x192x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x192x4xf32>) -> tensor<64x192x4x64xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x192x4x64xf32>
      mhlo.return %355 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x385x4x64xf32>) -> tensor<64x192x4x64xf32>
    %229 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%228, %arg125, %arg126, %218) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %230 = stablehlo.get_tuple_element %229[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %231 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %232 = "mhlo.fusion"(%231) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %233 = "mhlo.fusion"(%232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %234 = "mhlo.fusion"(%230, %233) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %235 = "mhlo.fusion"(%234) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %236 = "mhlo.fusion"(%235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %237 = "mhlo.fusion"(%236, %arg127, %arg128, %230, %233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      mhlo.return %343 : tensor<64x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x192x4x64xf32>
    %238 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%237, %arg129, %arg130) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>
    %239 = stablehlo.get_tuple_element %238[0] : (tuple<tensor<64x384x4x64xf32>, tensor<0xui8>>) -> tensor<64x384x4x64xf32>
    %240 = "mhlo.fusion"(%239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x384x4x64xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x384x4x64xf32>
      %338 = stablehlo.exponential %337 : tensor<64x384x4x64xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x384x4x64xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x384x4x64xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x384x4x64xf32>
      mhlo.return %341 : tensor<64x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x384x4x64xf32>) -> tensor<64x384x4x64xf32>
    %241 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%240, %arg131, %arg132, %230) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>) -> tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>
    %242 = stablehlo.get_tuple_element %241[0] : (tuple<tensor<64x192x4x64xf32>, tensor<0xui8>>) -> tensor<64x192x4x64xf32>
    %243 = "mhlo.fusion"(%242) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %337 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>) -> tensor<64x128xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %245 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %246 = "mhlo.fusion"(%242, %245) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x4x64xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x192x4x64xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x192x4x64xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x128x384xf32>, tensor<f32>) -> tensor<64x128xf32>
      mhlo.return %340 : tensor<64x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<64x128xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x128xf32>) -> tensor<64xf32>
    %248 = "mhlo.fusion"(%247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %339 = stablehlo.add %337, %338 : tensor<64xf32>
      %340 = stablehlo.rsqrt %339 : tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %249 = "mhlo.fusion"(%248, %arg133, %arg134, %242, %245) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<192xf32>, %arg247: tensor<192xf32>, %arg248: tensor<64x192x4x64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %337 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %338 = stablehlo.subtract %arg248, %337 : tensor<64x192x4x64xf32>
      %339 = stablehlo.broadcast_in_dim %arg245, dims = [0] : (tensor<64xf32>) -> tensor<64x192x4x64xf32>
      %340 = stablehlo.multiply %338, %339 : tensor<64x192x4x64xf32>
      %341 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<64x192x4x64xf32>
      %342 = stablehlo.multiply %340, %341 : tensor<64x192x4x64xf32>
      %343 = stablehlo.add %336, %342 : tensor<64x192x4x64xf32>
      %344 = mhlo.bitcast %343 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x192x4x64xf32>) -> tensor<64x192x2x2x8x8xf32>
      %345 = stablehlo.transpose %344, dims = [2, 4, 3, 5, 0, 1] : (tensor<64x192x2x2x8x8xf32>) -> tensor<2x8x2x8x64x192xf32>
      mhlo.return %345 : tensor<2x8x2x8x64x192xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<192xf32>, tensor<192xf32>, tensor<64x192x4x64xf32>, tensor<64xf32>) -> tensor<2x8x2x8x64x192xf32>
    %250 = "mhlo.fusion"(%3, %249) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x192x16x16xf32>, %arg246: tensor<2x8x2x8x64x192xf32>):
      %336 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %337 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %338 = stablehlo.multiply %336, %337 : tensor<8xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %340 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %341 = stablehlo.add %339, %340 : tensor<2x8xi32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %343 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %344 = stablehlo.compare LT, %342, %343 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %345 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %346 = stablehlo.add %342, %345 : tensor<1x1x2x8xi32>
      %347 = stablehlo.select %344, %346, %342 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %350 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %351 = stablehlo.compare LT, %341, %350 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %352 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %353 = stablehlo.add %341, %352 : tensor<2x8xi32>
      %354 = stablehlo.select %351, %353, %341 : tensor<2x8xi1>, tensor<2x8xi32>
      %355 = stablehlo.broadcast_in_dim %354, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %356 = stablehlo.concatenate %349, %355, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %358 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x64x192xf32>) -> tensor<256x64x192x1x1xf32>
      %359 = "stablehlo.scatter"(%arg245, %357, %358) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %360 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %360 : tensor<f32>
      }) : (tensor<64x192x16x16xf32>, tensor<256x2xi32>, tensor<256x64x192x1x1xf32>) -> tensor<64x192x16x16xf32>
      mhlo.return %359 : tensor<64x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x192x16x16xf32>, tensor<2x8x2x8x64x192xf32>) -> tensor<64x192x16x16xf32>
    %251 = stablehlo.custom_call @__cudnn$convForward(%250, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x192x16x16xf32>, tensor<384x192x1x1xf32>) -> tuple<tensor<64x384x16x16xf32>, tensor<0xui8>>
    %252 = stablehlo.get_tuple_element %251[0] : (tuple<tensor<64x384x16x16xf32>, tensor<0xui8>>) -> tensor<64x384x16x16xf32>
    %253 = "mhlo.fusion"(%2, %arg137, %arg136, %252, %arg232) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<384xf32>, %arg248: tensor<64x384x16x16xf32>, %arg249: tensor<384xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x384x16x16xf32>
      %338 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x384x16x16xf32>
      %340 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x384x16x16xf32>
      %342 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<64x384x16x16xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x384x16x16xf32>
      mhlo.return %343 : tensor<64x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<64x384x16x16xf32>, tensor<384xf32>) -> tensor<64x384x16x16xf32>
    %254 = stablehlo.custom_call @__cudnn$convForward(%253, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,768,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x384x16x16xf32>, tensor<768x384x1x1xf32>) -> tuple<tensor<64x768x16x16xf32>, tensor<0xui8>>
    %255 = stablehlo.get_tuple_element %254[0] : (tuple<tensor<64x768x16x16xf32>, tensor<0xui8>>) -> tensor<64x768x16x16xf32>
    %256 = "mhlo.fusion"(%arg140, %arg139, %1, %255, %arg234) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<64x768x16x16xf32>, %arg249: tensor<768xf32>):
      %336 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
      %337 = stablehlo.subtract %arg248, %336 : tensor<64x768x16x16xf32>
      %338 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
      %339 = stablehlo.multiply %337, %338 : tensor<64x768x16x16xf32>
      %340 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
      %341 = stablehlo.multiply %339, %340 : tensor<64x768x16x16xf32>
      %342 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<64x768x16x16xf32>
      %343 = stablehlo.add %341, %342 : tensor<64x768x16x16xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %344 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x768x16x16xf32>
      %345 = stablehlo.negate %343 : tensor<64x768x16x16xf32>
      %346 = stablehlo.exponential %345 : tensor<64x768x16x16xf32>
      %347 = stablehlo.add %344, %346 : tensor<64x768x16x16xf32>
      %348 = stablehlo.divide %344, %347 : tensor<64x768x16x16xf32>
      %349 = stablehlo.multiply %343, %348 : tensor<64x768x16x16xf32>
      mhlo.return %349 : tensor<64x768x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<64x768x16x16xf32>, tensor<768xf32>) -> tensor<64x768x16x16xf32>
    %257 = stablehlo.custom_call @__cudnn$convForward(%256, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,768,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<64x768x16x16xf32>, tensor<768x1x3x3xf32>) -> tuple<tensor<64x768x8x8xf32>, tensor<0xui8>>
    %258 = stablehlo.get_tuple_element %257[0] : (tuple<tensor<64x768x8x8xf32>, tensor<0xui8>>) -> tensor<64x768x8x8xf32>
    %259 = "mhlo.fusion"(%258, %arg143, %arg142, %arg236, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x768x8x8xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %336 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x768x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %339 = stablehlo.add %arg249, %338 : tensor<768xf32>
      %340 = stablehlo.rsqrt %339 : tensor<768xf32>
      %341 = stablehlo.broadcast_in_dim %340, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
      %342 = stablehlo.multiply %337, %341 : tensor<64x768x8x8xf32>
      %343 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
      %344 = stablehlo.multiply %342, %343 : tensor<64x768x8x8xf32>
      %345 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<64x768x8x8xf32>
      %346 = stablehlo.add %344, %345 : tensor<64x768x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x768x8x8xf32>
      %348 = stablehlo.negate %346 : tensor<64x768x8x8xf32>
      %349 = stablehlo.exponential %348 : tensor<64x768x8x8xf32>
      %350 = stablehlo.add %347, %349 : tensor<64x768x8x8xf32>
      %351 = stablehlo.divide %347, %350 : tensor<64x768x8x8xf32>
      %352 = stablehlo.multiply %346, %351 : tensor<64x768x8x8xf32>
      mhlo.return %352 : tensor<64x768x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x768x8x8xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<64x768x8x8xf32>
    %260 = stablehlo.custom_call @__cudnn$convForward(%259, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<64x768x8x8xf32>, tensor<512x768x1x1xf32>) -> tuple<tensor<64x512x8x8xf32>, tensor<0xui8>>
    %261 = stablehlo.get_tuple_element %260[0] : (tuple<tensor<64x512x8x8xf32>, tensor<0xui8>>) -> tensor<64x512x8x8xf32>
    %262 = "mhlo.fusion"(%arg146, %arg145, %261, %arg238, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<64x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %336 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %337 = stablehlo.subtract %arg247, %336 : tensor<64x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %339 = stablehlo.add %arg249, %338 : tensor<512xf32>
      %340 = stablehlo.rsqrt %339 : tensor<512xf32>
      %341 = stablehlo.broadcast_in_dim %340, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %342 = stablehlo.multiply %337, %341 : tensor<64x512x8x8xf32>
      %343 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %344 = stablehlo.multiply %342, %343 : tensor<64x512x8x8xf32>
      %345 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %346 = stablehlo.add %344, %345 : tensor<64x512x8x8xf32>
      mhlo.return %346 : tensor<64x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %263 = stablehlo.custom_call @__cudnn$convForward(%262, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x8x8xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<64x512x8x8xf32>, tensor<0xui8>>
    %264 = stablehlo.get_tuple_element %263[0] : (tuple<tensor<64x512x8x8xf32>, tensor<0xui8>>) -> tensor<64x512x8x8xf32>
    %265 = "mhlo.fusion"(%arg149, %arg148, %264, %arg240, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<64x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %336 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %337 = stablehlo.subtract %arg247, %336 : tensor<64x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %339 = stablehlo.add %arg249, %338 : tensor<512xf32>
      %340 = stablehlo.rsqrt %339 : tensor<512xf32>
      %341 = stablehlo.broadcast_in_dim %340, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %342 = stablehlo.multiply %337, %341 : tensor<64x512x8x8xf32>
      %343 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %344 = stablehlo.multiply %342, %343 : tensor<64x512x8x8xf32>
      %345 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %346 = stablehlo.add %344, %345 : tensor<64x512x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %347 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x512x8x8xf32>
      %348 = stablehlo.negate %346 : tensor<64x512x8x8xf32>
      %349 = stablehlo.exponential %348 : tensor<64x512x8x8xf32>
      %350 = stablehlo.add %347, %349 : tensor<64x512x8x8xf32>
      %351 = stablehlo.divide %347, %350 : tensor<64x512x8x8xf32>
      %352 = stablehlo.multiply %346, %351 : tensor<64x512x8x8xf32>
      mhlo.return %352 : tensor<64x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512x8x8xf32>
    %266 = stablehlo.custom_call @__cudnn$convForward(%265, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x8x8xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<64x256x8x8xf32>, tensor<0xui8>>
    %267 = stablehlo.get_tuple_element %266[0] : (tuple<tensor<64x256x8x8xf32>, tensor<0xui8>>) -> tensor<64x256x8x8xf32>
    %268 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x256x8x8xf32>):
      %336 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %337 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %338 = stablehlo.multiply %336, %337 : tensor<4xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %340 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %341 = stablehlo.add %339, %340 : tensor<2x4xi32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %343 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %344 = stablehlo.compare LT, %342, %343 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %345 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %346 = stablehlo.add %342, %345 : tensor<1x1x2x4xi32>
      %347 = stablehlo.select %344, %346, %342 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %350 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %351 = stablehlo.compare LT, %341, %350 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %352 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %353 = stablehlo.add %341, %352 : tensor<2x4xi32>
      %354 = stablehlo.select %351, %353, %341 : tensor<2x4xi1>, tensor<2x4xi32>
      %355 = stablehlo.broadcast_in_dim %354, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %356 = stablehlo.concatenate %349, %355, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %358 = "stablehlo.gather"(%arg245, %357) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x64x256x1x1xf32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x64x256x1x1xf32>) -> tensor<2x4x2x4x64x256xf32>
      %360 = stablehlo.transpose %359, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x64x256xf32>) -> tensor<64x256x2x2x4x4xf32>
      mhlo.return %360 : tensor<64x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8x8xf32>) -> tensor<64x256x2x2x4x4xf32>
    %269 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x8x8xf32>):
      %336 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %337 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %338 = stablehlo.multiply %336, %337 : tensor<4xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %340 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %341 = stablehlo.add %339, %340 : tensor<2x4xi32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %343 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %344 = stablehlo.compare LT, %342, %343 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %345 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %346 = stablehlo.add %342, %345 : tensor<1x1x2x4xi32>
      %347 = stablehlo.select %344, %346, %342 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %350 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %351 = stablehlo.compare LT, %341, %350 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %352 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %353 = stablehlo.add %341, %352 : tensor<2x4xi32>
      %354 = stablehlo.select %351, %353, %341 : tensor<2x4xi1>, tensor<2x4xi32>
      %355 = stablehlo.broadcast_in_dim %354, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %356 = stablehlo.concatenate %349, %355, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %358 = "stablehlo.gather"(%arg245, %357) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 64, 256, 1, 1>}> : (tensor<64x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x64x256x1x1xf32>
      %359 = mhlo.bitcast %358 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x64x256x1x1xf32>) -> tensor<64x64x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %360 = stablehlo.reduce(%359 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x64x256xf32>, tensor<f32>) -> tensor<64x64xf32>
      mhlo.return %360 : tensor<64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8x8xf32>) -> tensor<64x64xf32>
    %270 = "mhlo.fusion"(%269) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x64xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %336 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x64xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %336 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x64xf32>) -> tensor<64xf32>
    %271 = "mhlo.fusion"(%270) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %272 = "mhlo.fusion"(%268, %271) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x2x2x4x4xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x2x2x4x4xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x2x2x4x4xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x2x2x4x4xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x256x2x2x4x4xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x2x2x4x4xf32>, tensor<64xf32>) -> tensor<64xf32>
    %273 = "mhlo.fusion"(%arg151, %arg152, %268, %271, %272) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x2x2x4x4xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = mhlo.bitcast %336 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<1x64x16384xf32>
      %338 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x2x2x4x4xf32>
      %339 = stablehlo.subtract %arg247, %338 : tensor<64x256x2x2x4x4xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %340 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %341 = stablehlo.multiply %arg249, %340 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %342 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %343 = stablehlo.add %341, %342 : tensor<64xf32>
      %344 = stablehlo.rsqrt %343 : tensor<64xf32>
      %345 = stablehlo.broadcast_in_dim %344, dims = [0] : (tensor<64xf32>) -> tensor<64x256x2x2x4x4xf32>
      %346 = stablehlo.multiply %339, %345 : tensor<64x256x2x2x4x4xf32>
      %347 = mhlo.bitcast %346 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x256x2x2x4x4xf32>) -> tensor<1x64x16384xf32>
      %348 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %349 = mhlo.bitcast %348 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<1x64x16384xf32>
      %350 = stablehlo.multiply %347, %349 : tensor<1x64x16384xf32>
      %351 = stablehlo.add %337, %350 : tensor<1x64x16384xf32>
      mhlo.return %351 : tensor<1x64x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x2x2x4x4xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x16384xf32>
    %274 = mhlo.bitcast %273 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x64x16384xf32>) -> tensor<64x256x4x16xf32>
    %275 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%274, %arg153, %arg154) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<64x513x4x16xf32>, tensor<0xui8>>
    %276 = stablehlo.get_tuple_element %275[0] : (tuple<tensor<64x513x4x16xf32>, tensor<0xui8>>) -> tensor<64x513x4x16xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x513x4x16xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 257:513, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x256x4x16xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:257, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x1x4x16xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x16xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x16xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x16xf32>) -> tensor<64x256x4x16xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x256x4x16xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x256x4x16xf32>, tensor<f32>) -> tensor<64x256x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x256x4xf32>) -> tensor<64x256x4x16xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x256x4x16xf32>
      mhlo.return %355 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %278 = mhlo.bitcast %268 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<64x256x2x2x4x4xf32>) -> tensor<64x256x4x16xf32>
    %279 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%277, %arg155, %arg156, %278) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>) -> tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>
    %280 = stablehlo.get_tuple_element %279[0] : (tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>) -> tensor<64x256x4x16xf32>
    %281 = "mhlo.fusion"(%280) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>) -> tensor<64xf32>
    %282 = "mhlo.fusion"(%281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %283 = "mhlo.fusion"(%280, %282) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x4x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x4x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>, tensor<64xf32>) -> tensor<64xf32>
    %284 = "mhlo.fusion"(%arg157, %arg158, %280, %282, %283) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x4x16xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.subtract %arg247, %337 : tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.multiply %arg249, %339 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %342 = stablehlo.add %340, %341 : tensor<64xf32>
      %343 = stablehlo.rsqrt %342 : tensor<64xf32>
      %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %345 = stablehlo.multiply %338, %344 : tensor<64x256x4x16xf32>
      %346 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<64x256x4x16xf32>
      %348 = stablehlo.add %336, %347 : tensor<64x256x4x16xf32>
      mhlo.return %348 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %285 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%284, %arg159, %arg160) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<64x512x4x16xf32>, tensor<0xui8>>
    %286 = stablehlo.get_tuple_element %285[0] : (tuple<tensor<64x512x4x16xf32>, tensor<0xui8>>) -> tensor<64x512x4x16xf32>
    %287 = "mhlo.fusion"(%286) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x4x16xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x512x4x16xf32>
      %338 = stablehlo.exponential %337 : tensor<64x512x4x16xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x512x4x16xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x512x4x16xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x512x4x16xf32>
      mhlo.return %341 : tensor<64x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x4x16xf32>) -> tensor<64x512x4x16xf32>
    %288 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%287, %arg161, %arg162, %280) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>) -> tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>
    %289 = stablehlo.get_tuple_element %288[0] : (tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>) -> tensor<64x256x4x16xf32>
    %290 = "mhlo.fusion"(%289) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>) -> tensor<64xf32>
    %291 = "mhlo.fusion"(%290) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %292 = "mhlo.fusion"(%289, %291) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x4x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x4x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>, tensor<64xf32>) -> tensor<64xf32>
    %293 = "mhlo.fusion"(%arg163, %arg164, %289, %291, %292) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x4x16xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.subtract %arg247, %337 : tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.multiply %arg249, %339 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %342 = stablehlo.add %340, %341 : tensor<64xf32>
      %343 = stablehlo.rsqrt %342 : tensor<64xf32>
      %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %345 = stablehlo.multiply %338, %344 : tensor<64x256x4x16xf32>
      %346 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<64x256x4x16xf32>
      %348 = stablehlo.add %336, %347 : tensor<64x256x4x16xf32>
      mhlo.return %348 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %294 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%293, %arg165, %arg166) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<64x513x4x16xf32>, tensor<0xui8>>
    %295 = stablehlo.get_tuple_element %294[0] : (tuple<tensor<64x513x4x16xf32>, tensor<0xui8>>) -> tensor<64x513x4x16xf32>
    %296 = "mhlo.fusion"(%295) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x513x4x16xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 257:513, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x256x4x16xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:257, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x1x4x16xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x16xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x16xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x16xf32>) -> tensor<64x256x4x16xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x256x4x16xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x256x4x16xf32>, tensor<f32>) -> tensor<64x256x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x256x4xf32>) -> tensor<64x256x4x16xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x256x4x16xf32>
      mhlo.return %355 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %297 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%296, %arg167, %arg168, %289) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>) -> tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>
    %298 = stablehlo.get_tuple_element %297[0] : (tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>) -> tensor<64x256x4x16xf32>
    %299 = "mhlo.fusion"(%298) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>) -> tensor<64xf32>
    %300 = "mhlo.fusion"(%299) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %301 = "mhlo.fusion"(%298, %300) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x4x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x4x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>, tensor<64xf32>) -> tensor<64xf32>
    %302 = "mhlo.fusion"(%arg169, %arg170, %298, %300, %301) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x4x16xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.subtract %arg247, %337 : tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.multiply %arg249, %339 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %342 = stablehlo.add %340, %341 : tensor<64xf32>
      %343 = stablehlo.rsqrt %342 : tensor<64xf32>
      %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %345 = stablehlo.multiply %338, %344 : tensor<64x256x4x16xf32>
      %346 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<64x256x4x16xf32>
      %348 = stablehlo.add %336, %347 : tensor<64x256x4x16xf32>
      mhlo.return %348 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %303 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%302, %arg171, %arg172) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<64x512x4x16xf32>, tensor<0xui8>>
    %304 = stablehlo.get_tuple_element %303[0] : (tuple<tensor<64x512x4x16xf32>, tensor<0xui8>>) -> tensor<64x512x4x16xf32>
    %305 = "mhlo.fusion"(%304) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x4x16xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x512x4x16xf32>
      %338 = stablehlo.exponential %337 : tensor<64x512x4x16xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x512x4x16xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x512x4x16xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x512x4x16xf32>
      mhlo.return %341 : tensor<64x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x4x16xf32>) -> tensor<64x512x4x16xf32>
    %306 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%305, %arg173, %arg174, %298) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>) -> tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>
    %307 = stablehlo.get_tuple_element %306[0] : (tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>) -> tensor<64x256x4x16xf32>
    %308 = "mhlo.fusion"(%307) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>) -> tensor<64xf32>
    %309 = "mhlo.fusion"(%308) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %310 = "mhlo.fusion"(%307, %309) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x4x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x4x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>, tensor<64xf32>) -> tensor<64xf32>
    %311 = "mhlo.fusion"(%arg175, %arg176, %307, %309, %310) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x4x16xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.subtract %arg247, %337 : tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.multiply %arg249, %339 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %342 = stablehlo.add %340, %341 : tensor<64xf32>
      %343 = stablehlo.rsqrt %342 : tensor<64xf32>
      %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %345 = stablehlo.multiply %338, %344 : tensor<64x256x4x16xf32>
      %346 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<64x256x4x16xf32>
      %348 = stablehlo.add %336, %347 : tensor<64x256x4x16xf32>
      mhlo.return %348 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %312 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%311, %arg177, %arg178) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<64x513x4x16xf32>, tensor<0xui8>>
    %313 = stablehlo.get_tuple_element %312[0] : (tuple<tensor<64x513x4x16xf32>, tensor<0xui8>>) -> tensor<64x513x4x16xf32>
    %314 = "mhlo.fusion"(%313) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x513x4x16xf32>):
      %336 = stablehlo.slice %arg245 [0:64, 257:513, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.maximum %336, %337 : tensor<64x256x4x16xf32>
      %339 = stablehlo.slice %arg245 [0:64, 1:257, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
      %340 = stablehlo.slice %arg245 [0:64, 0:1, 0:4, 0:16] : (tensor<64x513x4x16xf32>) -> tensor<64x1x4x16xf32>
      %341 = mhlo.bitcast %340 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %342 = stablehlo.reduce(%341 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
      %343 = stablehlo.broadcast_in_dim %342, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
      %344 = stablehlo.subtract %340, %343 : tensor<64x1x4x16xf32>
      %345 = stablehlo.exponential %344 : tensor<64x1x4x16xf32>
      %346 = mhlo.bitcast %345 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %347 = stablehlo.reduce(%346 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x4x16xf32>, tensor<f32>) -> tensor<64x4xf32>
      %348 = stablehlo.broadcast_in_dim %347, dims = [0, 2] : (tensor<64x4xf32>) -> tensor<64x1x4x16xf32>
      %349 = stablehlo.divide %345, %348 : tensor<64x1x4x16xf32>
      %350 = mhlo.bitcast %349 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x1x4x16xf32>) -> tensor<64x4x16xf32>
      %351 = stablehlo.broadcast_in_dim %350, dims = [0, 2, 3] : (tensor<64x4x16xf32>) -> tensor<64x256x4x16xf32>
      %352 = stablehlo.multiply %339, %351 : tensor<64x256x4x16xf32>
      %353 = stablehlo.reduce(%352 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<64x256x4x16xf32>, tensor<f32>) -> tensor<64x256x4xf32>
      %354 = stablehlo.broadcast_in_dim %353, dims = [0, 1, 2] : (tensor<64x256x4xf32>) -> tensor<64x256x4x16xf32>
      %355 = stablehlo.multiply %338, %354 : tensor<64x256x4x16xf32>
      mhlo.return %355 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x513x4x16xf32>) -> tensor<64x256x4x16xf32>
    %315 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%314, %arg179, %arg180, %307) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>) -> tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>
    %316 = stablehlo.get_tuple_element %315[0] : (tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>) -> tensor<64x256x4x16xf32>
    %317 = "mhlo.fusion"(%316) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>) -> tensor<64xf32>
    %318 = "mhlo.fusion"(%317) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %319 = "mhlo.fusion"(%316, %318) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x4x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x4x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>, tensor<64xf32>) -> tensor<64xf32>
    %320 = "mhlo.fusion"(%arg181, %arg182, %316, %318, %319) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x4x16xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.subtract %arg247, %337 : tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.multiply %arg249, %339 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %342 = stablehlo.add %340, %341 : tensor<64xf32>
      %343 = stablehlo.rsqrt %342 : tensor<64xf32>
      %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %345 = stablehlo.multiply %338, %344 : tensor<64x256x4x16xf32>
      %346 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<64x256x4x16xf32>
      %348 = stablehlo.add %336, %347 : tensor<64x256x4x16xf32>
      mhlo.return %348 : tensor<64x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64x256x4x16xf32>
    %321 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%320, %arg183, %arg184) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<64x512x4x16xf32>, tensor<0xui8>>
    %322 = stablehlo.get_tuple_element %321[0] : (tuple<tensor<64x512x4x16xf32>, tensor<0xui8>>) -> tensor<64x512x4x16xf32>
    %323 = "mhlo.fusion"(%322) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512x4x16xf32>
      %337 = stablehlo.negate %arg245 : tensor<64x512x4x16xf32>
      %338 = stablehlo.exponential %337 : tensor<64x512x4x16xf32>
      %339 = stablehlo.add %336, %338 : tensor<64x512x4x16xf32>
      %340 = stablehlo.divide %336, %339 : tensor<64x512x4x16xf32>
      %341 = stablehlo.multiply %arg245, %340 : tensor<64x512x4x16xf32>
      mhlo.return %341 : tensor<64x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512x4x16xf32>) -> tensor<64x512x4x16xf32>
    %324 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%323, %arg185, %arg186, %316) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<64x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>) -> tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>
    %325 = stablehlo.get_tuple_element %324[0] : (tuple<tensor<64x256x4x16xf32>, tensor<0xui8>>) -> tensor<64x256x4x16xf32>
    %326 = "mhlo.fusion"(%325) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>):
      %336 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>) -> tensor<64xf32>
    %327 = "mhlo.fusion"(%326) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64xf32>
      mhlo.return %337 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %328 = "mhlo.fusion"(%325, %327) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x4x16xf32>, %arg246: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.subtract %arg245, %336 : tensor<64x256x4x16xf32>
      %338 = stablehlo.multiply %337, %337 : tensor<64x256x4x16xf32>
      %339 = mhlo.bitcast %338 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %340 = stablehlo.reduce(%339 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<64x16384xf32>, tensor<f32>) -> tensor<64xf32>
      mhlo.return %340 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x4x16xf32>, tensor<64xf32>) -> tensor<64xf32>
    %329 = "mhlo.fusion"(%arg187, %arg188, %325, %327, %328) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<64x256x4x16xf32>, %arg248: tensor<64xf32>, %arg249: tensor<64xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %337 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %338 = stablehlo.subtract %arg247, %337 : tensor<64x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %339 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %340 = stablehlo.multiply %arg249, %339 : tensor<64xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %341 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %342 = stablehlo.add %340, %341 : tensor<64xf32>
      %343 = stablehlo.rsqrt %342 : tensor<64xf32>
      %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<64xf32>) -> tensor<64x256x4x16xf32>
      %345 = stablehlo.multiply %338, %344 : tensor<64x256x4x16xf32>
      %346 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x4x16xf32>
      %347 = stablehlo.multiply %345, %346 : tensor<64x256x4x16xf32>
      %348 = stablehlo.add %336, %347 : tensor<64x256x4x16xf32>
      %349 = mhlo.bitcast %348 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x256x4x16xf32>) -> tensor<64x256x2x2x4x4xf32>
      %350 = stablehlo.transpose %349, dims = [2, 4, 3, 5, 0, 1] : (tensor<64x256x2x2x4x4xf32>) -> tensor<2x4x2x4x64x256xf32>
      mhlo.return %350 : tensor<2x4x2x4x64x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<64x256x4x16xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x2x4x64x256xf32>
    %330 = "mhlo.fusion"(%0, %329) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x256x8x8xf32>, %arg246: tensor<2x4x2x4x64x256xf32>):
      %336 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %337 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %338 = stablehlo.multiply %336, %337 : tensor<4xi32>
      %339 = stablehlo.broadcast_in_dim %338, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %340 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %341 = stablehlo.add %339, %340 : tensor<2x4xi32>
      %342 = mhlo.bitcast %341 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %343 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %344 = stablehlo.compare LT, %342, %343 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %345 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %346 = stablehlo.add %342, %345 : tensor<1x1x2x4xi32>
      %347 = stablehlo.select %344, %346, %342 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %348 = mhlo.bitcast %347 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %349 = stablehlo.broadcast_in_dim %348, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %350 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %351 = stablehlo.compare LT, %341, %350 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %352 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %353 = stablehlo.add %341, %352 : tensor<2x4xi32>
      %354 = stablehlo.select %351, %353, %341 : tensor<2x4xi1>, tensor<2x4xi32>
      %355 = stablehlo.broadcast_in_dim %354, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %356 = stablehlo.concatenate %349, %355, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %357 = mhlo.bitcast %356 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %358 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x64x256xf32>) -> tensor<64x64x256x1x1xf32>
      %359 = "stablehlo.scatter"(%arg245, %357, %358) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %360 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %360 : tensor<f32>
      }) : (tensor<64x256x8x8xf32>, tensor<64x2xi32>, tensor<64x64x256x1x1xf32>) -> tensor<64x256x8x8xf32>
      mhlo.return %359 : tensor<64x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x256x8x8xf32>, tensor<2x4x2x4x64x256xf32>) -> tensor<64x256x8x8xf32>
    %331 = stablehlo.custom_call @__cudnn$convForward(%330, %arg189) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[64,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<64x256x8x8xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<64x512x8x8xf32>, tensor<0xui8>>
    %332 = stablehlo.get_tuple_element %331[0] : (tuple<tensor<64x512x8x8xf32>, tensor<0xui8>>) -> tensor<64x512x8x8xf32>
    %333 = "mhlo.fusion"(%arg191, %arg190, %332, %arg242, %arg243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<64x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %336 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %337 = stablehlo.subtract %arg247, %336 : tensor<64x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %338 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %339 = stablehlo.add %arg249, %338 : tensor<512xf32>
      %340 = stablehlo.rsqrt %339 : tensor<512xf32>
      %341 = stablehlo.broadcast_in_dim %340, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %342 = stablehlo.multiply %337, %341 : tensor<64x512x8x8xf32>
      %343 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %344 = stablehlo.multiply %342, %343 : tensor<64x512x8x8xf32>
      %345 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<64x512x8x8xf32>
      %346 = stablehlo.add %344, %345 : tensor<64x512x8x8xf32>
      %347 = mhlo.bitcast %346 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<64x512x8x8xf32>) -> tensor<64x512x64xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.reduce(%347 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x512x64xf32>, tensor<f32>) -> tensor<64x512xf32>
      mhlo.return %348 : tensor<64x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<64x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<64x512xf32>
    %334 = "mhlo.fusion"(%333, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<64x512xf32>, %arg246: tensor<2x512xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %336 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x512xf32>
      %337 = stablehlo.multiply %arg245, %336 : tensor<64x512xf32>
      %338 = stablehlo.dot_general %337, %arg246, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<2x512xf32>) -> tensor<64x2xf32>
      mhlo.return %338 : tensor<64x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x512xf32>, tensor<2x512xf32>) -> tensor<64x2xf32>
    %335 = "mhlo.fusion"(%334, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64x2xf32>, %arg246: tensor<2xf32>):
      %336 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<2xf32>) -> tensor<64x2xf32>
      %337 = stablehlo.add %336, %arg245 : tensor<64x2xf32>
      mhlo.return %337 : tensor<64x2xf32>
    }) {output_operand_aliasing = []} : (tensor<64x2xf32>, tensor<2xf32>) -> tensor<64x2xf32>
    return %335 : tensor<64x2xf32>
  }
}
