module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<16x512xf32>, %arg1: tensor<2x512xf32>) -> tensor<16x2xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16x512xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x512xf32>, tensor<2x512xf32>) -> tensor<16x2xf32>
    return %2 : tensor<16x2xf32>
  }
  func.func private @region_1.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<16x192x4x64xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<16x192x4x64xf32>) -> tensor<16x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %1 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<16x192x2x2x8x8xf32>, %arg1: tensor<16xf32>) -> tensor<16x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x192x2x2x8x8xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<16x192x2x2x8x8xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<16x192x2x2x8x8xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
    return %6 : tensor<16x128xf32>
  }
  func.func private @region_1.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16xf32>) -> tensor<16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %4 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<16x128x4x256xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %1 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16xf32>) -> tensor<16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %4 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<16x128x4x256xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %1 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16xf32>) -> tensor<16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %4 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<16x128x4x256xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %1 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<16x128x4x256xf32>, %arg1: tensor<16xf32>) -> tensor<16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x128x4x256xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x128x4x256xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %4 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<16x128x4x256xf32>) -> tensor<16x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %1 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<16x128x2x2x16x16xf32>, %arg1: tensor<16xf32>) -> tensor<16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x128x2x2x16x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x128x2x2x16x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x128x2x2x16x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
    return %4 : tensor<16x256xf32>
  }
  func.func private @region_1.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<16x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<16x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x512x8x8xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x512x8x8xf32>) -> tensor<16x512x64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x512x64xf32>, tensor<f32>) -> tensor<16x512xf32>
    return %12 : tensor<16x512xf32>
  }
  func.func private @region_1.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<16x256x4x16xf32>) -> tensor<16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<16x256x4x16xf32>) -> tensor<16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<16x256x4x16xf32>) -> tensor<16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<16x256x4x16xf32>) -> tensor<16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<16x256x4x16xf32>) -> tensor<16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<16x256x4x16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x4x16xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x4x16xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<16x256x4x16xf32>) -> tensor<16xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<16x256x2x2x4x4xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<16xf32>) -> tensor<16x256x2x2x4x4xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x256x2x2x4x4xf32>
    %2 = stablehlo.multiply %1, %1 : tensor<16x256x2x2x4x4xf32>
    %3 = mhlo.bitcast %2 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @region_1.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<16x256x8x8xf32>) -> tensor<64x16xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x16x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<64x16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16x256xf32>, tensor<f32>) -> tensor<64x16xf32>
    return %24 : tensor<64x16xf32>
  }
  func.func private @region_1.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<16x192x16x16xf32>) -> tensor<256x16xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 192, 1, 1>}> : (tensor<16x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x16x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<256x16x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16x192xf32>, tensor<f32>) -> tensor<256x16xf32>
    return %24 : tensor<256x16xf32>
  }
  func.func private @region_1.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<16x128x32x32xf32>) -> tensor<1024x16xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 128, 1, 1>}> : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x16x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<1024x16x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x16x128xf32>, tensor<f32>) -> tensor<1024x16xf32>
    return %21 : tensor<1024x16xf32>
  }
  func.func private @region_1.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter(%arg0: tensor<16x256x8x8xf32>, %arg1: tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x8x8xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x16x256xf32>) -> tensor<64x16x256x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<16x256x8x8xf32>, tensor<64x2xi32>, tensor<64x16x256x1x1xf32>) -> tensor<16x256x8x8xf32>
    return %23 : tensor<16x256x8x8xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x4x16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<2x4x2x4x16x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.add %4, %5 : tensor<16xf32>
    %7 = stablehlo.rsqrt %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<16x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<16x256x4x16xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x256x2x2x4x4xf32>
    %14 = stablehlo.transpose %13, dims = [2, 4, 3, 5, 0, 1] : (tensor<16x256x2x2x4x4xf32>) -> tensor<2x4x2x4x16x256xf32>
    return %14 : tensor<2x4x2x4x16x256xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<16x256x8x8xf32>) -> tensor<16x256x2x2x4x4xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x16x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
    return %24 : tensor<16x256x2x2x4x4xf32>
  }
  func.func private @region_1.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.1(%arg0: tensor<16x192x16x16xf32>, %arg1: tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x16x16xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x16x192xf32>) -> tensor<256x16x192x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<16x192x16x16xf32>, tensor<256x2xi32>, tensor<256x16x192x1x1xf32>) -> tensor<16x192x16x16xf32>
    return %23 : tensor<16x192x16x16xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<2x8x2x8x16x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x192x2x2x8x8xf32>
    %15 = stablehlo.transpose %14, dims = [2, 4, 3, 5, 0, 1] : (tensor<16x192x2x2x8x8xf32>) -> tensor<2x8x2x8x16x192xf32>
    return %15 : tensor<2x8x2x8x16x192xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<16x192x16x16xf32>) -> tensor<16x192x2x2x8x8xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 192, 1, 1>}> : (tensor<16x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x16x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
    return %24 : tensor<16x192x2x2x8x8xf32>
  }
  func.func private @region_1.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.2(%arg0: tensor<16x128x32x32xf32>, %arg1: tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x32x32xf32> {
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
    %19 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x16x128xf32>) -> tensor<1024x16x128x1x1xf32>
    %20 = "stablehlo.scatter"(%arg0, %18, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %21 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }) : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x128x32x32xf32>
    return %20 : tensor<16x128x32x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<16xf32>) -> tensor<2x16x2x16x16x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<16x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<16x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<16x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<16x128x4x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x128x2x2x16x16xf32>
    %9 = stablehlo.transpose %8, dims = [2, 4, 3, 5, 0, 1] : (tensor<16x128x2x2x16x16xf32>) -> tensor<2x16x2x16x16x128xf32>
    return %9 : tensor<2x16x2x16x16x128xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<16x128x32x32xf32>) -> tensor<16x128x2x2x16x16xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 128, 1, 1>}> : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x16x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
    %21 = stablehlo.transpose %20, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
    return %21 : tensor<16x128x2x2x16x16xf32>
  }
  func.func private @fused_add(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x2x2x16x16xf32>, %arg4: tensor<16xf32>) -> tensor<1x16x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
    %2 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<16xf32>) -> tensor<16x128x2x2x16x16xf32>
    %3 = stablehlo.subtract %arg3, %2 : tensor<16x128x2x2x16x16xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<16xf32>) -> tensor<16x128x2x2x16x16xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x2x2x16x16xf32>
    %6 = mhlo.bitcast %5 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<1x16x131072xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %8 = mhlo.bitcast %7 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
    %9 = stablehlo.multiply %6, %8 : tensor<1x16x131072xf32>
    %10 = stablehlo.add %1, %9 : tensor<1x16x131072xf32>
    return %10 : tensor<1x16x131072xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x2x2x8x8xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<1x16x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.multiply %arg4, %2 : tensor<16xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16x192x2x2x8x8xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<16x192x2x2x8x8xf32>
    %6 = stablehlo.multiply %arg3, %2 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %8 = stablehlo.add %6, %7 : tensor<16xf32>
    %9 = stablehlo.rsqrt %8 : tensor<16xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<16xf32>) -> tensor<16x192x2x2x8x8xf32>
    %11 = stablehlo.multiply %5, %10 : tensor<16x192x2x2x8x8xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<1x16x49152xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
    %15 = stablehlo.multiply %12, %14 : tensor<1x16x49152xf32>
    %16 = stablehlo.add %1, %15 : tensor<1x16x49152xf32>
    return %16 : tensor<1x16x49152xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x2x2x4x4xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<1x16x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x2x2x4x4xf32>
    %3 = stablehlo.subtract %arg2, %2 : tensor<16x256x2x2x4x4xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %5 = stablehlo.multiply %arg4, %4 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x256x2x2x4x4xf32>
    %10 = stablehlo.multiply %3, %9 : tensor<16x256x2x2x4x4xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<1x16x16384xf32>
    %12 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %13 = mhlo.bitcast %12 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
    %14 = stablehlo.multiply %11, %13 : tensor<1x16x16384xf32>
    %15 = stablehlo.add %1, %14 : tensor<1x16x16384xf32>
    return %15 : tensor<1x16x16384xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x128x128x128xf32>, %arg4: tensor<128xf32>) -> tensor<16x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<16x128x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<16x128x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x128x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x128x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x128x128x128xf32>
    return %13 : tensor<16x128x128x128xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<16x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<16x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<16x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<16x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x64x128x128xf32>
    return %13 : tensor<16x64x128x128xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<16x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<16x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<16x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<16x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x64x128x128xf32>
    return %13 : tensor<16x64x128x128xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<16x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x256x64x64xf32>
    return %13 : tensor<16x256x64x64xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<16x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x256x64x64xf32>
    return %13 : tensor<16x256x64x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<16x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x256x64x64xf32>
    return %13 : tensor<16x256x64x64xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<16x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<16x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x64x128x128xf32>
    return %7 : tensor<16x64x128x128xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<16x32x128x128xf32>, %arg4: tensor<32xf32>) -> tensor<16x32x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x32x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x32x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x32x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x32x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<16x32x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<16x32x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x32x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x32x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x32x128x128xf32>
    return %13 : tensor<16x32x128x128xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<16x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<16x128x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<16x128x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x128x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x128x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x128x64x64xf32>
    return %13 : tensor<16x128x64x64xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<16x512x32x32xf32>, %arg4: tensor<512xf32>) -> tensor<16x512x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x512x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x512x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x512x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x512x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<16x512x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<16x512x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x512x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x512x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x512x32x32xf32>
    return %13 : tensor<16x512x32x32xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<16x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<16x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x64x64xf32>
    return %7 : tensor<16x128x64x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<16x128x64x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<16x128x64x64xf32>, %arg5: tensor<128xf32>) -> tensor<16x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<16x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x128x64x64xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<16x128x64x64xf32>
    return %8 : tensor<16x128x64x64xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x32x32xf32>
    return %7 : tensor<16x256x32x32xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x32x32xf32>
    return %7 : tensor<16x256x32x32xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<16x256x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<16x256x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x256x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x256x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x256x32x32xf32>
    return %13 : tensor<16x256x32x32xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<16x256x32x32xf32>, %arg4: tensor<256xf32>) -> tensor<16x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<16x256x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<16x256x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<16x256x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<16x256x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<16x256x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<16x256x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<16x256x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<16x256x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<16x256x32x32xf32>
    return %13 : tensor<16x256x32x32xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<16x256x4x256xf32>) -> tensor<16x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<16x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x256x4x256xf32>
    return %5 : tensor<16x256x4x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<16x256x4x256xf32>) -> tensor<16x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<16x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x256x4x256xf32>
    return %5 : tensor<16x256x4x256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<16x768x16x16xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
    %1 = stablehlo.subtract %arg0, %0 : tensor<16x768x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x768x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x768x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x768x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x768x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<16x768x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<16x768x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<16x768x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<16x768x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<16x768x16x16xf32>
    return %16 : tensor<16x768x16x16xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<16x512x16x16xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<16x512x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x512x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x512x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x512x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x512x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<16x512x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<16x512x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<16x512x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<16x512x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<16x512x16x16xf32>
    return %16 : tensor<16x512x16x16xf32>
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
  func.func private @fused_computation.216(%arg0: tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x128x4x256xf32>
    return %19 : tensor<16x128x4x256xf32>
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
  func.func private @fused_computation.219(%arg0: tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x128x4x256xf32>
    return %19 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<16xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<16x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<16x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<16x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<16x128x4x256xf32>
    return %7 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<16xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<16x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<16x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<16x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<16x128x4x256xf32>
    return %7 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x128x4x256xf32>, %arg4: tensor<16xf32>) -> tensor<16x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %2 = stablehlo.subtract %arg3, %1 : tensor<16x128x4x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<16x128x4x256xf32>
    %5 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<16x128x4x256xf32>
    %7 = stablehlo.add %0, %6 : tensor<16x128x4x256xf32>
    return %7 : tensor<16x128x4x256xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<16x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<16x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x384x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x384x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<16x384x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<16x384x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<16x384x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<16x384x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<16x384x16x16xf32>
    return %16 : tensor<16x384x16x16xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<16x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<16x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x384x16x16xf32>
    return %10 : tensor<16x384x16x16xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<16x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<16x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x384x16x16xf32>
    return %10 : tensor<16x384x16x16xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<16x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x384x4x64xf32>
    return %5 : tensor<16x384x4x64xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<16x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x384x4x64xf32>
    return %5 : tensor<16x384x4x64xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<16x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x384x4x64xf32>
    return %5 : tensor<16x384x4x64xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<16x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x384x4x64xf32>
    return %5 : tensor<16x384x4x64xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<16x768x8x8xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<16x768x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x768x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x768x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x768x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x768x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x768x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<16x768x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<16x768x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<16x768x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<16x768x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<16x768x8x8xf32>
    return %16 : tensor<16x768x8x8xf32>
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
  func.func private @fused_computation.253(%arg0: tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x192x4x64xf32>
    return %19 : tensor<16x192x4x64xf32>
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
  func.func private @fused_computation.254(%arg0: tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x192x4x64xf32>
    return %19 : tensor<16x192x4x64xf32>
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
  func.func private @fused_computation.255(%arg0: tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x192x4x64xf32>
    return %19 : tensor<16x192x4x64xf32>
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
  func.func private @fused_computation.256(%arg0: tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x192x4x64xf32>
    return %19 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<16x192x4x64xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<16x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %7 = stablehlo.add %5, %6 : tensor<16xf32>
    %8 = stablehlo.rsqrt %7 : tensor<16xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<16x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<16x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<16x192x4x64xf32>
    return %13 : tensor<16x192x4x64xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<16x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<16x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x512x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<16x512x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<16x512x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<16x512x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<16x512x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<16x512x8x8xf32>
    return %16 : tensor<16x512x8x8xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<16x512x4x16xf32>) -> tensor<16x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<16x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x512x4x16xf32>
    return %5 : tensor<16x512x4x16xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<16x512x4x16xf32>) -> tensor<16x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<16x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x512x4x16xf32>
    return %5 : tensor<16x512x4x16xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<16x512x4x16xf32>) -> tensor<16x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<16x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<16x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<16x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<16x512x4x16xf32>
    return %5 : tensor<16x512x4x16xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<16x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<16x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<16x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<16x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<16x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<16x512x8x8xf32>
    return %10 : tensor<16x512x8x8xf32>
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
  func.func private @fused_computation.314(%arg0: tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x256x4x16xf32>
    return %19 : tensor<16x256x4x16xf32>
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
  func.func private @fused_computation.315(%arg0: tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x256x4x16xf32>
    return %19 : tensor<16x256x4x16xf32>
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
  func.func private @fused_computation.316(%arg0: tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<16x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<16x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<16x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<16x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<16x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<16x256x4x16xf32>
    return %19 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x4x16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.add %4, %5 : tensor<16xf32>
    %7 = stablehlo.rsqrt %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<16x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<16x256x4x16xf32>
    return %12 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x4x16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.add %4, %5 : tensor<16xf32>
    %7 = stablehlo.rsqrt %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<16x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<16x256x4x16xf32>
    return %12 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x4x16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.add %4, %5 : tensor<16xf32>
    %7 = stablehlo.rsqrt %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<16x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<16x256x4x16xf32>
    return %12 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x4x16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.add %4, %5 : tensor<16xf32>
    %7 = stablehlo.rsqrt %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<16x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<16x256x4x16xf32>
    return %12 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<16x256x4x16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %2 = stablehlo.subtract %arg2, %1 : tensor<16x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %4 = stablehlo.multiply %arg4, %3 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %6 = stablehlo.add %4, %5 : tensor<16xf32>
    %7 = stablehlo.rsqrt %6 : tensor<16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %9 = stablehlo.multiply %2, %8 : tensor<16x256x4x16xf32>
    %10 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<16x256x4x16xf32>
    %12 = stablehlo.add %0, %11 : tensor<16x256x4x16xf32>
    return %12 : tensor<16x256x4x16xf32>
  }
  func.func private @fused_rsqrt(%arg0: tensor<512xf32>) -> tensor<512xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<512xf32>
    %2 = stablehlo.rsqrt %1 : tensor<512xf32>
    return %2 : tensor<512xf32>
  }
  func.func private @fused_rsqrt.1(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.2(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.3(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.4(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @fused_rsqrt.5(%arg0: tensor<256xf32>) -> tensor<256xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<256xf32>
    %2 = stablehlo.rsqrt %1 : tensor<256xf32>
    return %2 : tensor<256xf32>
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
  func.func private @fused_rsqrt.8(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.9(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.10(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.11(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.12(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.13(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.14(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.15(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_add.26(%arg0: tensor<16x2xf32>, %arg1: tensor<2xf32>) -> tensor<16x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<16x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<16x2xf32>
    return %1 : tensor<16x2xf32>
  }
  func.func private @fused_rsqrt.16(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.add %1, %2 : tensor<16xf32>
    %4 = stablehlo.rsqrt %3 : tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @fused_rsqrt.17(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.add %1, %2 : tensor<16xf32>
    %4 = stablehlo.rsqrt %3 : tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @fused_rsqrt.18(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.add %1, %2 : tensor<16xf32>
    %4 = stablehlo.rsqrt %3 : tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @fused_rsqrt.19(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.add %1, %2 : tensor<16xf32>
    %4 = stablehlo.rsqrt %3 : tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @fused_rsqrt.20(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %3 = stablehlo.add %1, %2 : tensor<16xf32>
    %4 = stablehlo.rsqrt %3 : tensor<16xf32>
    return %4 : tensor<16xf32>
  }
  func.func private @fused_multiply.25(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.26(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.27(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.28(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.29(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.30(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.31(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.32(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.33(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.34(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.35(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_multiply.36(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func private @fused_broadcast() -> tensor<16x192x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x16x16xf32>
    return %0 : tensor<16x192x16x16xf32>
  }
  func.func private @fused_broadcast.1() -> tensor<16x128x32x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x32x32xf32>
    return %0 : tensor<16x128x32x32xf32>
  }
  func.func private @fused_broadcast.2() -> tensor<16x256x8x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x8x8xf32>
    return %0 : tensor<16x256x8x8xf32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<1024x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.52(%arg0: tensor<16x256xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.53(%arg0: tensor<256x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.54(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.55(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.56(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.57(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.58(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.59(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.60(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.61(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.62(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.63(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.64(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.65(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.66(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.67(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.68(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.69(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.70(%arg0: tensor<16x128xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func private @fused_reduce.71(%arg0: tensor<64x16xf32>) -> tensor<16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<16x3x256x256xf32>) -> tensor<16x2xf32> {
    %0 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x8x8xf32>
      mhlo.return %313 : tensor<16x256x8x8xf32>
    }) {output_operand_aliasing = []} : () -> tensor<16x256x8x8xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x16x16xf32>
      mhlo.return %313 : tensor<16x192x16x16xf32>
    }) {output_operand_aliasing = []} : () -> tensor<16x192x16x16xf32>
    %2 = "mhlo.fusion"(%arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<512xf32>
      %315 = stablehlo.rsqrt %314 : tensor<512xf32>
      mhlo.return %315 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %3 = "mhlo.fusion"(%arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %4 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x32x32xf32>
      mhlo.return %313 : tensor<16x128x32x32xf32>
    }) {output_operand_aliasing = []} : () -> tensor<16x128x32x32xf32>
    %5 = "mhlo.fusion"(%arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %6 = "mhlo.fusion"(%arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %7 = "mhlo.fusion"(%arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %8 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %9 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %10 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %11 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %12 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %13 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %14 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<64xf32>
      %315 = stablehlo.rsqrt %314 : tensor<64xf32>
      mhlo.return %315 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %15 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<32xf32>
      %315 = stablehlo.rsqrt %314 : tensor<32xf32>
      mhlo.return %315 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %16 = stablehlo.custom_call @__cudnn$convForward(%arg244, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<16x3x256x256xf32>, tensor<32x3x3x3xf32>) -> tuple<tensor<16x32x128x128xf32>, tensor<0xui8>>
    %17 = stablehlo.get_tuple_element %16[0] : (tuple<tensor<16x32x128x128xf32>, tensor<0xui8>>) -> tensor<16x32x128x128xf32>
    %18 = "mhlo.fusion"(%arg2, %arg1, %15, %17, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<16x32x128x128xf32>, %arg249: tensor<32xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x32x128x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x32x128x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x32x128x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<32xf32>) -> tensor<16x32x128x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x32x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32x128x128xf32>
      %322 = stablehlo.negate %320 : tensor<16x32x128x128xf32>
      %323 = stablehlo.exponential %322 : tensor<16x32x128x128xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x32x128x128xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x32x128x128xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x32x128x128xf32>
      mhlo.return %326 : tensor<16x32x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<16x32x128x128xf32>, tensor<32xf32>) -> tensor<16x32x128x128xf32>
    %19 = stablehlo.custom_call @__cudnn$convForward(%18, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<16x32x128x128xf32>, tensor<64x32x1x1xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<0xui8>>
    %20 = stablehlo.get_tuple_element %19[0] : (tuple<tensor<16x64x128x128xf32>, tensor<0xui8>>) -> tensor<16x64x128x128xf32>
    %21 = "mhlo.fusion"(%arg5, %arg4, %14, %20, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<16x64x128x128xf32>, %arg249: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x64x128x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x64x128x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x64x128x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
      %322 = stablehlo.negate %320 : tensor<16x64x128x128xf32>
      %323 = stablehlo.exponential %322 : tensor<16x64x128x128xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x64x128x128xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x64x128x128xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x64x128x128xf32>
      mhlo.return %326 : tensor<16x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x128x128xf32>, tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %22 = stablehlo.custom_call @__cudnn$convForward(%21, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<16x64x128x128xf32>, tensor<64x1x3x3xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<0xui8>>
    %23 = stablehlo.get_tuple_element %22[0] : (tuple<tensor<16x64x128x128xf32>, tensor<0xui8>>) -> tensor<16x64x128x128xf32>
    %24 = "mhlo.fusion"(%arg8, %arg7, %13, %23, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<16x64x128x128xf32>, %arg249: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x64x128x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x64x128x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x64x128x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x64x128x128xf32>
      %322 = stablehlo.negate %320 : tensor<16x64x128x128xf32>
      %323 = stablehlo.exponential %322 : tensor<16x64x128x128xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x64x128x128xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x64x128x128xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x64x128x128xf32>
      mhlo.return %326 : tensor<16x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x128x128xf32>, tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %25 = stablehlo.custom_call @__cudnn$convForward(%24, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<16x64x128x128xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<16x64x128x128xf32>, tensor<0xui8>>
    %26 = stablehlo.get_tuple_element %25[0] : (tuple<tensor<16x64x128x128xf32>, tensor<0xui8>>) -> tensor<16x64x128x128xf32>
    %27 = "mhlo.fusion"(%12, %arg11, %arg10, %26, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<16x64x128x128xf32>, %arg249: tensor<64xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x64x128x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x64x128x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x64x128x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<16x64x128x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x64x128x128xf32>
      mhlo.return %320 : tensor<16x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<16x64x128x128xf32>, tensor<64xf32>) -> tensor<16x64x128x128xf32>
    %28 = stablehlo.custom_call @__cudnn$convForward(%27, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<16x64x128x128xf32>, tensor<128x64x1x1xf32>) -> tuple<tensor<16x128x128x128xf32>, tensor<0xui8>>
    %29 = stablehlo.get_tuple_element %28[0] : (tuple<tensor<16x128x128x128xf32>, tensor<0xui8>>) -> tensor<16x128x128x128xf32>
    %30 = "mhlo.fusion"(%arg14, %arg13, %11, %29, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x128x128x128xf32>, %arg249: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x128x128x128xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x128x128x128xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x128x128x128xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x128x128xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x128x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x128x128xf32>
      %322 = stablehlo.negate %320 : tensor<16x128x128x128xf32>
      %323 = stablehlo.exponential %322 : tensor<16x128x128x128xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x128x128x128xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x128x128x128xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x128x128x128xf32>
      mhlo.return %326 : tensor<16x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x128x128xf32>, tensor<128xf32>) -> tensor<16x128x128x128xf32>
    %31 = stablehlo.custom_call @__cudnn$convForward(%30, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<0xui8>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<16x128x64x64xf32>, tensor<0xui8>>) -> tensor<16x128x64x64xf32>
    %33 = "mhlo.fusion"(%arg17, %arg16, %10, %32, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x128x64x64xf32>, %arg249: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x128x64x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x128x64x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x128x64x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x128x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x64x64xf32>
      %322 = stablehlo.negate %320 : tensor<16x128x64x64xf32>
      %323 = stablehlo.exponential %322 : tensor<16x128x64x64xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x128x64x64xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x128x64x64xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x128x64x64xf32>
      mhlo.return %326 : tensor<16x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x64x64xf32>, tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %34 = stablehlo.custom_call @__cudnn$convForward(%33, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x64x64xf32>, tensor<128x128x1x1xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<0xui8>>
    %35 = stablehlo.get_tuple_element %34[0] : (tuple<tensor<16x128x64x64xf32>, tensor<0xui8>>) -> tensor<16x128x64x64xf32>
    %36 = "mhlo.fusion"(%arg20, %arg19, %9, %35, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<16x128x64x64xf32>, %arg249: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x128x64x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x128x64x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x128x64x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x128x64x64xf32>
      mhlo.return %320 : tensor<16x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x64x64xf32>, tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %37 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<128xf32>
      %315 = stablehlo.rsqrt %314 : tensor<128xf32>
      mhlo.return %315 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %38 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %39 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %314 = stablehlo.add %arg245, %313 : tensor<256xf32>
      %315 = stablehlo.rsqrt %314 : tensor<256xf32>
      mhlo.return %315 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %40 = stablehlo.custom_call @__cudnn$convForward(%36, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<0xui8>>
    %41 = stablehlo.get_tuple_element %40[0] : (tuple<tensor<16x256x64x64xf32>, tensor<0xui8>>) -> tensor<16x256x64x64xf32>
    %42 = "mhlo.fusion"(%arg23, %arg22, %39, %41, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x64x64xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x64x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x64x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x64x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
      %322 = stablehlo.negate %320 : tensor<16x256x64x64xf32>
      %323 = stablehlo.exponential %322 : tensor<16x256x64x64xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x256x64x64xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x256x64x64xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x256x64x64xf32>
      mhlo.return %326 : tensor<16x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %43 = stablehlo.custom_call @__cudnn$convForward(%42, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<0xui8>>
    %44 = stablehlo.get_tuple_element %43[0] : (tuple<tensor<16x256x64x64xf32>, tensor<0xui8>>) -> tensor<16x256x64x64xf32>
    %45 = "mhlo.fusion"(%arg26, %arg25, %38, %44, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x64x64xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x64x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x64x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x64x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
      %322 = stablehlo.negate %320 : tensor<16x256x64x64xf32>
      %323 = stablehlo.exponential %322 : tensor<16x256x64x64xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x256x64x64xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x256x64x64xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x256x64x64xf32>
      mhlo.return %326 : tensor<16x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %46 = stablehlo.custom_call @__cudnn$convForward(%45, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<16x128x64x64xf32>, tensor<0xui8>>
    %47 = stablehlo.get_tuple_element %46[0] : (tuple<tensor<16x128x64x64xf32>, tensor<0xui8>>) -> tensor<16x128x64x64xf32>
    %48 = "mhlo.fusion"(%36, %arg29, %arg28, %37, %47, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x128x64x64xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<128xf32>, %arg249: tensor<16x128x64x64xf32>, %arg250: tensor<128xf32>):
      %313 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %314 = stablehlo.subtract %arg249, %313 : tensor<16x128x64x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x128x64x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x128x64x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x64x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x128x64x64xf32>
      %321 = stablehlo.add %arg245, %320 : tensor<16x128x64x64xf32>
      mhlo.return %321 : tensor<16x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<16x128x64x64xf32>, tensor<128xf32>) -> tensor<16x128x64x64xf32>
    %49 = stablehlo.custom_call @__cudnn$convForward(%48, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<16x256x64x64xf32>, tensor<0xui8>>
    %50 = stablehlo.get_tuple_element %49[0] : (tuple<tensor<16x256x64x64xf32>, tensor<0xui8>>) -> tensor<16x256x64x64xf32>
    %51 = "mhlo.fusion"(%arg32, %arg31, %8, %50, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x64x64xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x64x64xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x64x64xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x64x64xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x64x64xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x64x64xf32>
      %322 = stablehlo.negate %320 : tensor<16x256x64x64xf32>
      %323 = stablehlo.exponential %322 : tensor<16x256x64x64xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x256x64x64xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x256x64x64xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x256x64x64xf32>
      mhlo.return %326 : tensor<16x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x64x64xf32>, tensor<256xf32>) -> tensor<16x256x64x64xf32>
    %52 = stablehlo.custom_call @__cudnn$convForward(%51, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>
    %53 = stablehlo.get_tuple_element %52[0] : (tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>) -> tensor<16x256x32x32xf32>
    %54 = "mhlo.fusion"(%arg35, %arg34, %7, %53, %arg216) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x32x32xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x32x32xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x32x32xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x32x32xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
      %322 = stablehlo.negate %320 : tensor<16x256x32x32xf32>
      %323 = stablehlo.exponential %322 : tensor<16x256x32x32xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x256x32x32xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x256x32x32xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x256x32x32xf32>
      mhlo.return %326 : tensor<16x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>, tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %55 = stablehlo.custom_call @__cudnn$convForward(%54, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x32x32xf32>, tensor<256x256x1x1xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>
    %56 = stablehlo.get_tuple_element %55[0] : (tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>) -> tensor<16x256x32x32xf32>
    %57 = "mhlo.fusion"(%arg38, %arg37, %6, %56, %arg218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x32x32xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x32x32xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x32x32xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x32x32xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x32x32xf32>
      mhlo.return %320 : tensor<16x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>, tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %58 = stablehlo.custom_call @__cudnn$convForward(%57, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x32x32xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>
    %59 = stablehlo.get_tuple_element %58[0] : (tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>) -> tensor<16x256x32x32xf32>
    %60 = "mhlo.fusion"(%arg41, %arg40, %5, %59, %arg220) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x32x32xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x32x32xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x32x32xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x32x32xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x32x32xf32>
      %322 = stablehlo.negate %320 : tensor<16x256x32x32xf32>
      %323 = stablehlo.exponential %322 : tensor<16x256x32x32xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x256x32x32xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x256x32x32xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x256x32x32xf32>
      mhlo.return %326 : tensor<16x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>, tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %61 = stablehlo.custom_call @__cudnn$convForward(%60, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x32x32xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<16x128x32x32xf32>, tensor<0xui8>>
    %62 = stablehlo.get_tuple_element %61[0] : (tuple<tensor<16x128x32x32xf32>, tensor<0xui8>>) -> tensor<16x128x32x32xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x32x32xf32>):
      %313 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %314 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %315 = stablehlo.add %313, %314 : tensor<2x16xi32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %317 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %318 = stablehlo.compare LT, %316, %317 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %319 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %320 = stablehlo.add %316, %319 : tensor<1x1x2x16xi32>
      %321 = stablehlo.select %318, %320, %316 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %324 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %325 = stablehlo.compare LT, %315, %324 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %326 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %327 = stablehlo.add %315, %326 : tensor<2x16xi32>
      %328 = stablehlo.select %325, %327, %315 : tensor<2x16xi1>, tensor<2x16xi32>
      %329 = stablehlo.broadcast_in_dim %328, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %330 = stablehlo.concatenate %323, %329, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %331 = mhlo.bitcast %330 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %332 = "stablehlo.gather"(%arg245, %331) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 128, 1, 1>}> : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x16x128x1x1xf32>
      %333 = mhlo.bitcast %332 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<2x16x2x16x16x128xf32>
      %334 = stablehlo.transpose %333, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x2x2x16x16xf32>
      mhlo.return %334 : tensor<16x128x2x2x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x32x32xf32>) -> tensor<16x128x2x2x16x16xf32>
    %64 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x32x32xf32>):
      %313 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %314 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %315 = stablehlo.add %313, %314 : tensor<2x16xi32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %317 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %318 = stablehlo.compare LT, %316, %317 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %319 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %320 = stablehlo.add %316, %319 : tensor<1x1x2x16xi32>
      %321 = stablehlo.select %318, %320, %316 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %324 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %325 = stablehlo.compare LT, %315, %324 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %326 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %327 = stablehlo.add %315, %326 : tensor<2x16xi32>
      %328 = stablehlo.select %325, %327, %315 : tensor<2x16xi1>, tensor<2x16xi32>
      %329 = stablehlo.broadcast_in_dim %328, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %330 = stablehlo.concatenate %323, %329, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %331 = mhlo.bitcast %330 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %332 = "stablehlo.gather"(%arg245, %331) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 128, 1, 1>}> : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x16x128x1x1xf32>
      %333 = mhlo.bitcast %332 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x16x128x1x1xf32>) -> tensor<1024x16x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %334 = stablehlo.reduce(%333 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x16x128xf32>, tensor<f32>) -> tensor<1024x16xf32>
      mhlo.return %334 : tensor<1024x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x32x32xf32>) -> tensor<1024x16xf32>
    %65 = "mhlo.fusion"(%64) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x16xf32>) -> tensor<16xf32>
    %66 = "mhlo.fusion"(%65) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %67 = "mhlo.fusion"(%63, %66) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x2x2x16x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x128x2x2x16x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x128x2x2x16x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x128x2x2x16x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %317 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x2x2x16x16xf32>, tensor<16xf32>) -> tensor<16x256xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %69 = "mhlo.fusion"(%68) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %316 = stablehlo.add %314, %315 : tensor<16xf32>
      %317 = stablehlo.rsqrt %316 : tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %70 = "mhlo.fusion"(%arg44, %arg43, %69, %63, %66) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x2x2x16x16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
      %315 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<16xf32>) -> tensor<16x128x2x2x16x16xf32>
      %316 = stablehlo.subtract %arg248, %315 : tensor<16x128x2x2x16x16xf32>
      %317 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<16xf32>) -> tensor<16x128x2x2x16x16xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x128x2x2x16x16xf32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<1x16x131072xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %321 = mhlo.bitcast %320 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<1x16x131072xf32>
      %322 = stablehlo.multiply %319, %321 : tensor<1x16x131072xf32>
      %323 = stablehlo.add %314, %322 : tensor<1x16x131072xf32>
      mhlo.return %323 : tensor<1x16x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<16xf32>, tensor<16x128x2x2x16x16xf32>, tensor<16xf32>) -> tensor<1x16x131072xf32>
    %71 = mhlo.bitcast %70 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x131072xf32>) -> tensor<16x128x4x256xf32>
    %72 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%71, %arg45, %arg46) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<16x257x4x256xf32>, tensor<0xui8>>
    %73 = stablehlo.get_tuple_element %72[0] : (tuple<tensor<16x257x4x256xf32>, tensor<0xui8>>) -> tensor<16x257x4x256xf32>
    %74 = "mhlo.fusion"(%73) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x257x4x256xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x128x4x256xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x256xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x256xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x256xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x128x4x256xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x128x4x256xf32>
      mhlo.return %332 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %75 = mhlo.bitcast %63 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x128x2x2x16x16xf32>) -> tensor<16x128x4x256xf32>
    %76 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%74, %arg47, %arg48, %75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<16x128x4x256xf32>) -> tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>
    %77 = stablehlo.get_tuple_element %76[0] : (tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>) -> tensor<16x128x4x256xf32>
    %78 = "mhlo.fusion"(%77) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %314 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>) -> tensor<16x256xf32>
    %79 = "mhlo.fusion"(%78) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %80 = "mhlo.fusion"(%79) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %81 = "mhlo.fusion"(%77, %80) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x128x4x256xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x128x4x256xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %317 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x256xf32>
    %82 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %316 = stablehlo.add %314, %315 : tensor<16xf32>
      %317 = stablehlo.rsqrt %316 : tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %84 = "mhlo.fusion"(%arg50, %arg49, %83, %77, %80) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %315 = stablehlo.subtract %arg248, %314 : tensor<16x128x4x256xf32>
      %316 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<16x128x4x256xf32>
      %318 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<16x128x4x256xf32>
      %320 = stablehlo.add %313, %319 : tensor<16x128x4x256xf32>
      mhlo.return %320 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %85 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%84, %arg51, %arg52) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<16x256x4x256xf32>, tensor<0xui8>>
    %86 = stablehlo.get_tuple_element %85[0] : (tuple<tensor<16x256x4x256xf32>, tensor<0xui8>>) -> tensor<16x256x4x256xf32>
    %87 = "mhlo.fusion"(%86) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x256x4x256xf32>
      %315 = stablehlo.exponential %314 : tensor<16x256x4x256xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x256x4x256xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x256x4x256xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x256x4x256xf32>
      mhlo.return %318 : tensor<16x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x256xf32>) -> tensor<16x256x4x256xf32>
    %88 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%87, %arg53, %arg54, %77) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<16x128x4x256xf32>) -> tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>
    %89 = stablehlo.get_tuple_element %88[0] : (tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>) -> tensor<16x128x4x256xf32>
    %90 = "mhlo.fusion"(%89) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %314 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>) -> tensor<16x256xf32>
    %91 = "mhlo.fusion"(%90) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %92 = "mhlo.fusion"(%91) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %93 = "mhlo.fusion"(%89, %92) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x128x4x256xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x128x4x256xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %317 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x256xf32>
    %94 = "mhlo.fusion"(%93) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %95 = "mhlo.fusion"(%94) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %316 = stablehlo.add %314, %315 : tensor<16xf32>
      %317 = stablehlo.rsqrt %316 : tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %96 = "mhlo.fusion"(%arg56, %arg55, %95, %89, %92) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %315 = stablehlo.subtract %arg248, %314 : tensor<16x128x4x256xf32>
      %316 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<16x128x4x256xf32>
      %318 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<16x128x4x256xf32>
      %320 = stablehlo.add %313, %319 : tensor<16x128x4x256xf32>
      mhlo.return %320 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %97 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%96, %arg57, %arg58) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<16x257x4x256xf32>, tensor<0xui8>>
    %98 = stablehlo.get_tuple_element %97[0] : (tuple<tensor<16x257x4x256xf32>, tensor<0xui8>>) -> tensor<16x257x4x256xf32>
    %99 = "mhlo.fusion"(%98) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x257x4x256xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 129:257, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x128x4x256xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x128x4x256xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:129, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:256] : (tensor<16x257x4x256xf32>) -> tensor<16x1x4x256xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x256xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x256xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x256xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x256xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x256xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x256xf32>) -> tensor<16x4x256xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x256xf32>) -> tensor<16x128x4x256xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x128x4x256xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x128x4x256xf32>, tensor<f32>) -> tensor<16x128x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x128x4xf32>) -> tensor<16x128x4x256xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x128x4x256xf32>
      mhlo.return %332 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x257x4x256xf32>) -> tensor<16x128x4x256xf32>
    %100 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%99, %arg59, %arg60, %89) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<16x128x4x256xf32>) -> tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>
    %101 = stablehlo.get_tuple_element %100[0] : (tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>) -> tensor<16x128x4x256xf32>
    %102 = "mhlo.fusion"(%101) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %314 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>) -> tensor<16x256xf32>
    %103 = "mhlo.fusion"(%102) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %104 = "mhlo.fusion"(%103) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %105 = "mhlo.fusion"(%101, %104) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x128x4x256xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x128x4x256xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %317 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x256xf32>
    %106 = "mhlo.fusion"(%105) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %107 = "mhlo.fusion"(%106) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %316 = stablehlo.add %314, %315 : tensor<16xf32>
      %317 = stablehlo.rsqrt %316 : tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %108 = "mhlo.fusion"(%arg62, %arg61, %107, %101, %104) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %315 = stablehlo.subtract %arg248, %314 : tensor<16x128x4x256xf32>
      %316 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<16x128x4x256xf32>
      %318 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<16x128x4x256xf32>
      %320 = stablehlo.add %313, %319 : tensor<16x128x4x256xf32>
      mhlo.return %320 : tensor<16x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x128x4x256xf32>
    %109 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%108, %arg63, %arg64) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<16x256x4x256xf32>, tensor<0xui8>>
    %110 = stablehlo.get_tuple_element %109[0] : (tuple<tensor<16x256x4x256xf32>, tensor<0xui8>>) -> tensor<16x256x4x256xf32>
    %111 = "mhlo.fusion"(%110) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x256xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x256x4x256xf32>
      %315 = stablehlo.exponential %314 : tensor<16x256x4x256xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x256x4x256xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x256x4x256xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x256x4x256xf32>
      mhlo.return %318 : tensor<16x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x256xf32>) -> tensor<16x256x4x256xf32>
    %112 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%111, %arg65, %arg66, %101) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<16x128x4x256xf32>) -> tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>
    %113 = stablehlo.get_tuple_element %112[0] : (tuple<tensor<16x128x4x256xf32>, tensor<0xui8>>) -> tensor<16x128x4x256xf32>
    %114 = "mhlo.fusion"(%113) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %314 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>) -> tensor<16x256xf32>
    %115 = "mhlo.fusion"(%114) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %116 = "mhlo.fusion"(%115) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %117 = "mhlo.fusion"(%113, %116) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x4x256xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x128x4x256xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x128x4x256xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x256x512xf32>, tensor<f32>) -> tensor<16x256xf32>
      mhlo.return %317 : tensor<16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<16x256xf32>
    %118 = "mhlo.fusion"(%117) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256xf32>) -> tensor<16xf32>
    %119 = "mhlo.fusion"(%118) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %316 = stablehlo.add %314, %315 : tensor<16xf32>
      %317 = stablehlo.rsqrt %316 : tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %120 = "mhlo.fusion"(%arg68, %arg67, %119, %113, %116) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<16xf32>, %arg248: tensor<16x128x4x256xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %314 = stablehlo.broadcast_in_dim %arg249, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %315 = stablehlo.subtract %arg248, %314 : tensor<16x128x4x256xf32>
      %316 = stablehlo.broadcast_in_dim %arg247, dims = [0] : (tensor<16xf32>) -> tensor<16x128x4x256xf32>
      %317 = stablehlo.multiply %315, %316 : tensor<16x128x4x256xf32>
      %318 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<16x128x4x256xf32>
      %319 = stablehlo.multiply %317, %318 : tensor<16x128x4x256xf32>
      %320 = stablehlo.add %313, %319 : tensor<16x128x4x256xf32>
      %321 = mhlo.bitcast %320 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x128x4x256xf32>) -> tensor<16x128x2x2x16x16xf32>
      %322 = stablehlo.transpose %321, dims = [2, 4, 3, 5, 0, 1] : (tensor<16x128x2x2x16x16xf32>) -> tensor<2x16x2x16x16x128xf32>
      mhlo.return %322 : tensor<2x16x2x16x16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<16xf32>, tensor<16x128x4x256xf32>, tensor<16xf32>) -> tensor<2x16x2x16x16x128xf32>
    %121 = "mhlo.fusion"(%4, %120) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128x32x32xf32>, %arg246: tensor<2x16x2x16x16x128xf32>):
      %313 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %314 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %315 = stablehlo.add %313, %314 : tensor<2x16xi32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %317 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %318 = stablehlo.compare LT, %316, %317 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %319 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %320 = stablehlo.add %316, %319 : tensor<1x1x2x16xi32>
      %321 = stablehlo.select %318, %320, %316 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %324 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %325 = stablehlo.compare LT, %315, %324 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %326 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %327 = stablehlo.add %315, %326 : tensor<2x16xi32>
      %328 = stablehlo.select %325, %327, %315 : tensor<2x16xi1>, tensor<2x16xi32>
      %329 = stablehlo.broadcast_in_dim %328, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %330 = stablehlo.concatenate %323, %329, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %331 = mhlo.bitcast %330 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %332 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x16x128xf32>) -> tensor<1024x16x128x1x1xf32>
      %333 = "stablehlo.scatter"(%arg245, %331, %332) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %334 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %334 : tensor<f32>
      }) : (tensor<16x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x16x128x1x1xf32>) -> tensor<16x128x32x32xf32>
      mhlo.return %333 : tensor<16x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128x32x32xf32>, tensor<2x16x2x16x16x128xf32>) -> tensor<16x128x32x32xf32>
    %122 = stablehlo.custom_call @__cudnn$convForward(%121, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<16x128x32x32xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>
    %123 = stablehlo.get_tuple_element %122[0] : (tuple<tensor<16x256x32x32xf32>, tensor<0xui8>>) -> tensor<16x256x32x32xf32>
    %124 = "mhlo.fusion"(%arg71, %arg70, %3, %123, %arg222) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<16x256x32x32xf32>, %arg249: tensor<256xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x256x32x32xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x256x32x32xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x256x32x32xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x32x32xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x256x32x32xf32>
      mhlo.return %320 : tensor<16x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<16x256x32x32xf32>, tensor<256xf32>) -> tensor<16x256x32x32xf32>
    %125 = stablehlo.custom_call @__cudnn$convForward(%124, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x32x32xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<16x512x32x32xf32>, tensor<0xui8>>
    %126 = stablehlo.get_tuple_element %125[0] : (tuple<tensor<16x512x32x32xf32>, tensor<0xui8>>) -> tensor<16x512x32x32xf32>
    %127 = "mhlo.fusion"(%arg74, %arg73, %2, %126, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<16x512x32x32xf32>, %arg249: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
      %314 = stablehlo.subtract %arg248, %313 : tensor<16x512x32x32xf32>
      %315 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
      %316 = stablehlo.multiply %314, %315 : tensor<16x512x32x32xf32>
      %317 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
      %318 = stablehlo.multiply %316, %317 : tensor<16x512x32x32xf32>
      %319 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<16x512x32x32xf32>
      %320 = stablehlo.add %318, %319 : tensor<16x512x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %321 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x32x32xf32>
      %322 = stablehlo.negate %320 : tensor<16x512x32x32xf32>
      %323 = stablehlo.exponential %322 : tensor<16x512x32x32xf32>
      %324 = stablehlo.add %321, %323 : tensor<16x512x32x32xf32>
      %325 = stablehlo.divide %321, %324 : tensor<16x512x32x32xf32>
      %326 = stablehlo.multiply %320, %325 : tensor<16x512x32x32xf32>
      mhlo.return %326 : tensor<16x512x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<16x512x32x32xf32>, tensor<512xf32>) -> tensor<16x512x32x32xf32>
    %128 = stablehlo.custom_call @__cudnn$convForward(%127, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x32x32xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<16x512x16x16xf32>, tensor<0xui8>>
    %129 = stablehlo.get_tuple_element %128[0] : (tuple<tensor<16x512x16x16xf32>, tensor<0xui8>>) -> tensor<16x512x16x16xf32>
    %130 = "mhlo.fusion"(%arg77, %arg76, %129, %arg226, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<16x512x16x16xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x512x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<512xf32>
      %317 = stablehlo.rsqrt %316 : tensor<512xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x512x16x16xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x512x16x16xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<16x512x16x16xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x512x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512x16x16xf32>
      %325 = stablehlo.negate %323 : tensor<16x512x16x16xf32>
      %326 = stablehlo.exponential %325 : tensor<16x512x16x16xf32>
      %327 = stablehlo.add %324, %326 : tensor<16x512x16x16xf32>
      %328 = stablehlo.divide %324, %327 : tensor<16x512x16x16xf32>
      %329 = stablehlo.multiply %323, %328 : tensor<16x512x16x16xf32>
      mhlo.return %329 : tensor<16x512x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<16x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<16x512x16x16xf32>
    %131 = stablehlo.custom_call @__cudnn$convForward(%130, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x16x16xf32>, tensor<384x512x1x1xf32>) -> tuple<tensor<16x384x16x16xf32>, tensor<0xui8>>
    %132 = stablehlo.get_tuple_element %131[0] : (tuple<tensor<16x384x16x16xf32>, tensor<0xui8>>) -> tensor<16x384x16x16xf32>
    %133 = "mhlo.fusion"(%arg80, %arg79, %132, %arg228, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<16x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<384xf32>
      %317 = stablehlo.rsqrt %316 : tensor<384xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x384x16x16xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x384x16x16xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x384x16x16xf32>
      mhlo.return %323 : tensor<16x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<16x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %134 = stablehlo.custom_call @__cudnn$convForward(%133, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x16x16xf32>, tensor<384x1x3x3xf32>) -> tuple<tensor<16x384x16x16xf32>, tensor<0xui8>>
    %135 = stablehlo.get_tuple_element %134[0] : (tuple<tensor<16x384x16x16xf32>, tensor<0xui8>>) -> tensor<16x384x16x16xf32>
    %136 = "mhlo.fusion"(%arg83, %arg82, %135, %arg230, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<16x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<384xf32>
      %317 = stablehlo.rsqrt %316 : tensor<384xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x384x16x16xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x384x16x16xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x384x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x384x16x16xf32>
      %325 = stablehlo.negate %323 : tensor<16x384x16x16xf32>
      %326 = stablehlo.exponential %325 : tensor<16x384x16x16xf32>
      %327 = stablehlo.add %324, %326 : tensor<16x384x16x16xf32>
      %328 = stablehlo.divide %324, %327 : tensor<16x384x16x16xf32>
      %329 = stablehlo.multiply %323, %328 : tensor<16x384x16x16xf32>
      mhlo.return %329 : tensor<16x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<16x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %137 = stablehlo.custom_call @__cudnn$convForward(%136, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x16x16xf32>, tensor<192x384x1x1xf32>) -> tuple<tensor<16x192x16x16xf32>, tensor<0xui8>>
    %138 = stablehlo.get_tuple_element %137[0] : (tuple<tensor<16x192x16x16xf32>, tensor<0xui8>>) -> tensor<16x192x16x16xf32>
    %139 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x16x16xf32>):
      %313 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %315 = stablehlo.multiply %313, %314 : tensor<8xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %317 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %318 = stablehlo.add %316, %317 : tensor<2x8xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %320 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %321 = stablehlo.compare LT, %319, %320 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %322 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %323 = stablehlo.add %319, %322 : tensor<1x1x2x8xi32>
      %324 = stablehlo.select %321, %323, %319 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %327 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %328 = stablehlo.compare LT, %318, %327 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %329 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %330 = stablehlo.add %318, %329 : tensor<2x8xi32>
      %331 = stablehlo.select %328, %330, %318 : tensor<2x8xi1>, tensor<2x8xi32>
      %332 = stablehlo.broadcast_in_dim %331, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %333 = stablehlo.concatenate %326, %332, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %334 = mhlo.bitcast %333 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %335 = "stablehlo.gather"(%arg245, %334) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 192, 1, 1>}> : (tensor<16x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x16x192x1x1xf32>
      %336 = mhlo.bitcast %335 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<2x8x2x8x16x192xf32>
      %337 = stablehlo.transpose %336, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x2x2x8x8xf32>
      mhlo.return %337 : tensor<16x192x2x2x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x16x16xf32>) -> tensor<16x192x2x2x8x8xf32>
    %140 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x16x16xf32>):
      %313 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %315 = stablehlo.multiply %313, %314 : tensor<8xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %317 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %318 = stablehlo.add %316, %317 : tensor<2x8xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %320 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %321 = stablehlo.compare LT, %319, %320 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %322 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %323 = stablehlo.add %319, %322 : tensor<1x1x2x8xi32>
      %324 = stablehlo.select %321, %323, %319 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %327 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %328 = stablehlo.compare LT, %318, %327 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %329 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %330 = stablehlo.add %318, %329 : tensor<2x8xi32>
      %331 = stablehlo.select %328, %330, %318 : tensor<2x8xi1>, tensor<2x8xi32>
      %332 = stablehlo.broadcast_in_dim %331, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %333 = stablehlo.concatenate %326, %332, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %334 = mhlo.bitcast %333 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %335 = "stablehlo.gather"(%arg245, %334) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 192, 1, 1>}> : (tensor<16x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x16x192x1x1xf32>
      %336 = mhlo.bitcast %335 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x16x192x1x1xf32>) -> tensor<256x16x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x16x192xf32>, tensor<f32>) -> tensor<256x16xf32>
      mhlo.return %337 : tensor<256x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x16x16xf32>) -> tensor<256x16xf32>
    %141 = "mhlo.fusion"(%140) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<256x16xf32>) -> tensor<16xf32>
    %142 = "mhlo.fusion"(%139, %141) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x2x2x8x8xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x2x2x8x8xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x2x2x8x8xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x2x2x8x8xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x2x2x8x8xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %143 = "mhlo.fusion"(%142) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %144 = "mhlo.fusion"(%arg85, %arg86, %139, %143, %141) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x2x2x8x8xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %316 = stablehlo.multiply %arg249, %315 : tensor<16xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0] : (tensor<16xf32>) -> tensor<16x192x2x2x8x8xf32>
      %318 = stablehlo.subtract %arg247, %317 : tensor<16x192x2x2x8x8xf32>
      %319 = stablehlo.multiply %arg248, %315 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %320 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %321 = stablehlo.add %319, %320 : tensor<16xf32>
      %322 = stablehlo.rsqrt %321 : tensor<16xf32>
      %323 = stablehlo.broadcast_in_dim %322, dims = [0] : (tensor<16xf32>) -> tensor<16x192x2x2x8x8xf32>
      %324 = stablehlo.multiply %318, %323 : tensor<16x192x2x2x8x8xf32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<1x16x49152xf32>
      %326 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<1x16x49152xf32>
      %328 = stablehlo.multiply %325, %327 : tensor<1x16x49152xf32>
      %329 = stablehlo.add %314, %328 : tensor<1x16x49152xf32>
      mhlo.return %329 : tensor<1x16x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x2x2x8x8xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x16x49152xf32>
    %145 = mhlo.bitcast %144 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x49152xf32>) -> tensor<16x192x4x64xf32>
    %146 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%145, %arg87, %arg88) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>
    %147 = stablehlo.get_tuple_element %146[0] : (tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>) -> tensor<16x385x4x64xf32>
    %148 = "mhlo.fusion"(%147) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x385x4x64xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x192x4x64xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x64xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x64xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x64xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x192x4x64xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x192x4x64xf32>
      mhlo.return %332 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %149 = mhlo.bitcast %139 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x192x2x2x8x8xf32>) -> tensor<16x192x4x64xf32>
    %150 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%148, %arg89, %arg90, %149) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %151 = stablehlo.get_tuple_element %150[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %152 = "mhlo.fusion"(%151) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %153 = "mhlo.fusion"(%152) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %154 = "mhlo.fusion"(%151, %153) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %156 = "mhlo.fusion"(%arg91, %arg92, %151, %155, %153) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %157 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%156, %arg93, %arg94) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>
    %158 = stablehlo.get_tuple_element %157[0] : (tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>) -> tensor<16x384x4x64xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x384x4x64xf32>
      %315 = stablehlo.exponential %314 : tensor<16x384x4x64xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x384x4x64xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x384x4x64xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x384x4x64xf32>
      mhlo.return %318 : tensor<16x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32>
    %160 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%159, %arg95, %arg96, %151) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %161 = stablehlo.get_tuple_element %160[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %162 = "mhlo.fusion"(%161) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %163 = "mhlo.fusion"(%162) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %164 = "mhlo.fusion"(%161, %163) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %166 = "mhlo.fusion"(%arg97, %arg98, %161, %165, %163) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %167 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%166, %arg99, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>
    %168 = stablehlo.get_tuple_element %167[0] : (tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>) -> tensor<16x385x4x64xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x385x4x64xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x192x4x64xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x64xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x64xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x64xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x192x4x64xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x192x4x64xf32>
      mhlo.return %332 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %170 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%169, %arg101, %arg102, %161) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %171 = stablehlo.get_tuple_element %170[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %172 = "mhlo.fusion"(%171) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %173 = "mhlo.fusion"(%172) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %174 = "mhlo.fusion"(%171, %173) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %176 = "mhlo.fusion"(%arg103, %arg104, %171, %175, %173) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %177 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%176, %arg105, %arg106) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>
    %178 = stablehlo.get_tuple_element %177[0] : (tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>) -> tensor<16x384x4x64xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x384x4x64xf32>
      %315 = stablehlo.exponential %314 : tensor<16x384x4x64xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x384x4x64xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x384x4x64xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x384x4x64xf32>
      mhlo.return %318 : tensor<16x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32>
    %180 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%179, %arg107, %arg108, %171) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %181 = stablehlo.get_tuple_element %180[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %182 = "mhlo.fusion"(%181) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %183 = "mhlo.fusion"(%182) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %184 = "mhlo.fusion"(%181, %183) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %185 = "mhlo.fusion"(%184) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %186 = "mhlo.fusion"(%arg109, %arg110, %181, %185, %183) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %187 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%186, %arg111, %arg112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>
    %188 = stablehlo.get_tuple_element %187[0] : (tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>) -> tensor<16x385x4x64xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x385x4x64xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x192x4x64xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x64xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x64xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x64xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x192x4x64xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x192x4x64xf32>
      mhlo.return %332 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %190 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%189, %arg113, %arg114, %181) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %191 = stablehlo.get_tuple_element %190[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %192 = "mhlo.fusion"(%191) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %193 = "mhlo.fusion"(%192) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %194 = "mhlo.fusion"(%191, %193) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %196 = "mhlo.fusion"(%arg115, %arg116, %191, %195, %193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %197 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%196, %arg117, %arg118) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>
    %198 = stablehlo.get_tuple_element %197[0] : (tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>) -> tensor<16x384x4x64xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x384x4x64xf32>
      %315 = stablehlo.exponential %314 : tensor<16x384x4x64xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x384x4x64xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x384x4x64xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x384x4x64xf32>
      mhlo.return %318 : tensor<16x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32>
    %200 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%199, %arg119, %arg120, %191) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %201 = stablehlo.get_tuple_element %200[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %202 = "mhlo.fusion"(%201) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %203 = "mhlo.fusion"(%202) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %204 = "mhlo.fusion"(%201, %203) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %205 = "mhlo.fusion"(%204) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %206 = "mhlo.fusion"(%arg121, %arg122, %201, %205, %203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %207 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%206, %arg123, %arg124) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>
    %208 = stablehlo.get_tuple_element %207[0] : (tuple<tensor<16x385x4x64xf32>, tensor<0xui8>>) -> tensor<16x385x4x64xf32>
    %209 = "mhlo.fusion"(%208) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x385x4x64xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 193:385, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x192x4x64xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x192x4x64xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:193, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:64] : (tensor<16x385x4x64xf32>) -> tensor<16x1x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x64xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x64xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x64xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x64xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x64xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x64xf32>) -> tensor<16x4x64xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x64xf32>) -> tensor<16x192x4x64xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x192x4x64xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x192x4x64xf32>, tensor<f32>) -> tensor<16x192x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x192x4xf32>) -> tensor<16x192x4x64xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x192x4x64xf32>
      mhlo.return %332 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x385x4x64xf32>) -> tensor<16x192x4x64xf32>
    %210 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%209, %arg125, %arg126, %201) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %211 = stablehlo.get_tuple_element %210[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %212 = "mhlo.fusion"(%211) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %213 = "mhlo.fusion"(%212) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %214 = "mhlo.fusion"(%211, %213) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %215 = "mhlo.fusion"(%214) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %216 = "mhlo.fusion"(%arg127, %arg128, %211, %215, %213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      mhlo.return %326 : tensor<16x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x192x4x64xf32>
    %217 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%216, %arg129, %arg130) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>
    %218 = stablehlo.get_tuple_element %217[0] : (tuple<tensor<16x384x4x64xf32>, tensor<0xui8>>) -> tensor<16x384x4x64xf32>
    %219 = "mhlo.fusion"(%218) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x384x4x64xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x384x4x64xf32>
      %315 = stablehlo.exponential %314 : tensor<16x384x4x64xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x384x4x64xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x384x4x64xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x384x4x64xf32>
      mhlo.return %318 : tensor<16x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<16x384x4x64xf32>) -> tensor<16x384x4x64xf32>
    %220 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%219, %arg131, %arg132, %211) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>) -> tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>
    %221 = stablehlo.get_tuple_element %220[0] : (tuple<tensor<16x192x4x64xf32>, tensor<0xui8>>) -> tensor<16x192x4x64xf32>
    %222 = "mhlo.fusion"(%221) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %314 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>) -> tensor<16x128xf32>
    %223 = "mhlo.fusion"(%222) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %224 = "mhlo.fusion"(%221, %223) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x4x64xf32>, %arg246: tensor<16xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg246, %313 : tensor<16xf32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %316 = stablehlo.subtract %arg245, %315 : tensor<16x192x4x64xf32>
      %317 = stablehlo.multiply %316, %316 : tensor<16x192x4x64xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x128x384xf32>, tensor<f32>) -> tensor<16x128xf32>
      mhlo.return %319 : tensor<16x128xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x4x64xf32>, tensor<16xf32>) -> tensor<16x128xf32>
    %225 = "mhlo.fusion"(%224) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x128xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x128xf32>) -> tensor<16xf32>
    %226 = "mhlo.fusion"(%arg133, %arg134, %221, %225, %223) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<16x192x4x64xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %315 = stablehlo.multiply %arg249, %314 : tensor<16xf32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %317 = stablehlo.subtract %arg247, %316 : tensor<16x192x4x64xf32>
      %318 = stablehlo.multiply %arg248, %314 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x192x4x64xf32>
      %323 = stablehlo.multiply %317, %322 : tensor<16x192x4x64xf32>
      %324 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<16x192x4x64xf32>
      %325 = stablehlo.multiply %323, %324 : tensor<16x192x4x64xf32>
      %326 = stablehlo.add %313, %325 : tensor<16x192x4x64xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x192x4x64xf32>) -> tensor<16x192x2x2x8x8xf32>
      %328 = stablehlo.transpose %327, dims = [2, 4, 3, 5, 0, 1] : (tensor<16x192x2x2x8x8xf32>) -> tensor<2x8x2x8x16x192xf32>
      mhlo.return %328 : tensor<2x8x2x8x16x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<16x192x4x64xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<2x8x2x8x16x192xf32>
    %227 = "mhlo.fusion"(%1, %226) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x192x16x16xf32>, %arg246: tensor<2x8x2x8x16x192xf32>):
      %313 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %315 = stablehlo.multiply %313, %314 : tensor<8xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %317 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %318 = stablehlo.add %316, %317 : tensor<2x8xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %320 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %321 = stablehlo.compare LT, %319, %320 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %322 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %323 = stablehlo.add %319, %322 : tensor<1x1x2x8xi32>
      %324 = stablehlo.select %321, %323, %319 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %327 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %328 = stablehlo.compare LT, %318, %327 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %329 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %330 = stablehlo.add %318, %329 : tensor<2x8xi32>
      %331 = stablehlo.select %328, %330, %318 : tensor<2x8xi1>, tensor<2x8xi32>
      %332 = stablehlo.broadcast_in_dim %331, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %333 = stablehlo.concatenate %326, %332, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %334 = mhlo.bitcast %333 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %335 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x16x192xf32>) -> tensor<256x16x192x1x1xf32>
      %336 = "stablehlo.scatter"(%arg245, %334, %335) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %337 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %337 : tensor<f32>
      }) : (tensor<16x192x16x16xf32>, tensor<256x2xi32>, tensor<256x16x192x1x1xf32>) -> tensor<16x192x16x16xf32>
      mhlo.return %336 : tensor<16x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x192x16x16xf32>, tensor<2x8x2x8x16x192xf32>) -> tensor<16x192x16x16xf32>
    %228 = stablehlo.custom_call @__cudnn$convForward(%227, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x192x16x16xf32>, tensor<384x192x1x1xf32>) -> tuple<tensor<16x384x16x16xf32>, tensor<0xui8>>
    %229 = stablehlo.get_tuple_element %228[0] : (tuple<tensor<16x384x16x16xf32>, tensor<0xui8>>) -> tensor<16x384x16x16xf32>
    %230 = "mhlo.fusion"(%arg137, %arg136, %229, %arg232, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<16x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<384xf32>
      %317 = stablehlo.rsqrt %316 : tensor<384xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x384x16x16xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x384x16x16xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<16x384x16x16xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x384x16x16xf32>
      mhlo.return %323 : tensor<16x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<16x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<16x384x16x16xf32>
    %231 = stablehlo.custom_call @__cudnn$convForward(%230, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,768,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x384x16x16xf32>, tensor<768x384x1x1xf32>) -> tuple<tensor<16x768x16x16xf32>, tensor<0xui8>>
    %232 = stablehlo.get_tuple_element %231[0] : (tuple<tensor<16x768x16x16xf32>, tensor<0xui8>>) -> tensor<16x768x16x16xf32>
    %233 = "mhlo.fusion"(%232, %arg140, %arg139, %arg234, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x768x16x16xf32>, %arg246: tensor<768xf32>, %arg247: tensor<768xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x768x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<768xf32>
      %317 = stablehlo.rsqrt %316 : tensor<768xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x768x16x16xf32>
      %320 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x768x16x16xf32>
      %322 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<16x768x16x16xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x768x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x768x16x16xf32>
      %325 = stablehlo.negate %323 : tensor<16x768x16x16xf32>
      %326 = stablehlo.exponential %325 : tensor<16x768x16x16xf32>
      %327 = stablehlo.add %324, %326 : tensor<16x768x16x16xf32>
      %328 = stablehlo.divide %324, %327 : tensor<16x768x16x16xf32>
      %329 = stablehlo.multiply %323, %328 : tensor<16x768x16x16xf32>
      mhlo.return %329 : tensor<16x768x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x768x16x16xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<16x768x16x16xf32>
    %234 = stablehlo.custom_call @__cudnn$convForward(%233, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,768,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<16x768x16x16xf32>, tensor<768x1x3x3xf32>) -> tuple<tensor<16x768x8x8xf32>, tensor<0xui8>>
    %235 = stablehlo.get_tuple_element %234[0] : (tuple<tensor<16x768x8x8xf32>, tensor<0xui8>>) -> tensor<16x768x8x8xf32>
    %236 = "mhlo.fusion"(%arg143, %arg142, %235, %arg236, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<16x768x8x8xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x768x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<768xf32>
      %317 = stablehlo.rsqrt %316 : tensor<768xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x768x8x8xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x768x8x8xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<16x768x8x8xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x768x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x768x8x8xf32>
      %325 = stablehlo.negate %323 : tensor<16x768x8x8xf32>
      %326 = stablehlo.exponential %325 : tensor<16x768x8x8xf32>
      %327 = stablehlo.add %324, %326 : tensor<16x768x8x8xf32>
      %328 = stablehlo.divide %324, %327 : tensor<16x768x8x8xf32>
      %329 = stablehlo.multiply %323, %328 : tensor<16x768x8x8xf32>
      mhlo.return %329 : tensor<16x768x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<16x768x8x8xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<16x768x8x8xf32>
    %237 = stablehlo.custom_call @__cudnn$convForward(%236, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<16x768x8x8xf32>, tensor<512x768x1x1xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<0xui8>>
    %238 = stablehlo.get_tuple_element %237[0] : (tuple<tensor<16x512x8x8xf32>, tensor<0xui8>>) -> tensor<16x512x8x8xf32>
    %239 = "mhlo.fusion"(%arg146, %arg145, %238, %arg238, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<16x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<512xf32>
      %317 = stablehlo.rsqrt %316 : tensor<512xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x512x8x8xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x512x8x8xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x512x8x8xf32>
      mhlo.return %323 : tensor<16x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %240 = stablehlo.custom_call @__cudnn$convForward(%239, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x8x8xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<0xui8>>
    %241 = stablehlo.get_tuple_element %240[0] : (tuple<tensor<16x512x8x8xf32>, tensor<0xui8>>) -> tensor<16x512x8x8xf32>
    %242 = "mhlo.fusion"(%arg149, %arg148, %241, %arg240, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<16x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<512xf32>
      %317 = stablehlo.rsqrt %316 : tensor<512xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x512x8x8xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x512x8x8xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x512x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %324 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512x8x8xf32>
      %325 = stablehlo.negate %323 : tensor<16x512x8x8xf32>
      %326 = stablehlo.exponential %325 : tensor<16x512x8x8xf32>
      %327 = stablehlo.add %324, %326 : tensor<16x512x8x8xf32>
      %328 = stablehlo.divide %324, %327 : tensor<16x512x8x8xf32>
      %329 = stablehlo.multiply %323, %328 : tensor<16x512x8x8xf32>
      mhlo.return %329 : tensor<16x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<16x512x8x8xf32>
    %243 = stablehlo.custom_call @__cudnn$convForward(%242, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x8x8xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<16x256x8x8xf32>, tensor<0xui8>>
    %244 = stablehlo.get_tuple_element %243[0] : (tuple<tensor<16x256x8x8xf32>, tensor<0xui8>>) -> tensor<16x256x8x8xf32>
    %245 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x256x8x8xf32>):
      %313 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %315 = stablehlo.multiply %313, %314 : tensor<4xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %317 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %318 = stablehlo.add %316, %317 : tensor<2x4xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %320 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %321 = stablehlo.compare LT, %319, %320 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %322 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %323 = stablehlo.add %319, %322 : tensor<1x1x2x4xi32>
      %324 = stablehlo.select %321, %323, %319 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %327 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %328 = stablehlo.compare LT, %318, %327 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %329 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %330 = stablehlo.add %318, %329 : tensor<2x4xi32>
      %331 = stablehlo.select %328, %330, %318 : tensor<2x4xi1>, tensor<2x4xi32>
      %332 = stablehlo.broadcast_in_dim %331, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %333 = stablehlo.concatenate %326, %332, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %334 = mhlo.bitcast %333 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %335 = "stablehlo.gather"(%arg245, %334) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x16x256x1x1xf32>
      %336 = mhlo.bitcast %335 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<2x4x2x4x16x256xf32>
      %337 = stablehlo.transpose %336, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x2x2x4x4xf32>
      mhlo.return %337 : tensor<16x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8x8xf32>) -> tensor<16x256x2x2x4x4xf32>
    %246 = "mhlo.fusion"(%244) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x8x8xf32>):
      %313 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %315 = stablehlo.multiply %313, %314 : tensor<4xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %317 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %318 = stablehlo.add %316, %317 : tensor<2x4xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %320 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %321 = stablehlo.compare LT, %319, %320 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %322 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %323 = stablehlo.add %319, %322 : tensor<1x1x2x4xi32>
      %324 = stablehlo.select %321, %323, %319 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %327 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %328 = stablehlo.compare LT, %318, %327 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %329 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %330 = stablehlo.add %318, %329 : tensor<2x4xi32>
      %331 = stablehlo.select %328, %330, %318 : tensor<2x4xi1>, tensor<2x4xi32>
      %332 = stablehlo.broadcast_in_dim %331, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %333 = stablehlo.concatenate %326, %332, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %334 = mhlo.bitcast %333 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %335 = "stablehlo.gather"(%arg245, %334) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 16, 256, 1, 1>}> : (tensor<16x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x16x256x1x1xf32>
      %336 = mhlo.bitcast %335 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x16x256x1x1xf32>) -> tensor<64x16x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.reduce(%336 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x16x256xf32>, tensor<f32>) -> tensor<64x16xf32>
      mhlo.return %337 : tensor<64x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8x8xf32>) -> tensor<64x16xf32>
    %247 = "mhlo.fusion"(%246) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x16xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %313 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x16xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %313 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<64x16xf32>) -> tensor<16xf32>
    %248 = "mhlo.fusion"(%247) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %249 = "mhlo.fusion"(%245, %248) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x2x2x4x4xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x2x2x4x4xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x2x2x4x4xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x2x2x4x4xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x2x2x4x4xf32>, tensor<16xf32>) -> tensor<16xf32>
    %250 = "mhlo.fusion"(%arg151, %arg152, %245, %248, %249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x2x2x4x4xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = mhlo.bitcast %313 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
      %315 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x2x2x4x4xf32>
      %316 = stablehlo.subtract %arg247, %315 : tensor<16x256x2x2x4x4xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %318 = stablehlo.multiply %arg249, %317 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %319 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %320 = stablehlo.add %318, %319 : tensor<16xf32>
      %321 = stablehlo.rsqrt %320 : tensor<16xf32>
      %322 = stablehlo.broadcast_in_dim %321, dims = [0] : (tensor<16xf32>) -> tensor<16x256x2x2x4x4xf32>
      %323 = stablehlo.multiply %316, %322 : tensor<16x256x2x2x4x4xf32>
      %324 = mhlo.bitcast %323 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<1x16x16384xf32>
      %325 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %326 = mhlo.bitcast %325 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<1x16x16384xf32>
      %327 = stablehlo.multiply %324, %326 : tensor<1x16x16384xf32>
      %328 = stablehlo.add %314, %327 : tensor<1x16x16384xf32>
      mhlo.return %328 : tensor<1x16x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x2x2x4x4xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x16x16384xf32>
    %251 = mhlo.bitcast %250 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x16x16384xf32>) -> tensor<16x256x4x16xf32>
    %252 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%251, %arg153, %arg154) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<16x513x4x16xf32>, tensor<0xui8>>
    %253 = stablehlo.get_tuple_element %252[0] : (tuple<tensor<16x513x4x16xf32>, tensor<0xui8>>) -> tensor<16x513x4x16xf32>
    %254 = "mhlo.fusion"(%253) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x513x4x16xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x256x4x16xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x16xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x16xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x16xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x256x4x16xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x256x4x16xf32>
      mhlo.return %332 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %255 = mhlo.bitcast %245 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<16x256x2x2x4x4xf32>) -> tensor<16x256x4x16xf32>
    %256 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%254, %arg155, %arg156, %255) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>) -> tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>
    %257 = stablehlo.get_tuple_element %256[0] : (tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>) -> tensor<16x256x4x16xf32>
    %258 = "mhlo.fusion"(%257) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>) -> tensor<16xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %260 = "mhlo.fusion"(%257, %259) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x4x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x4x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %261 = "mhlo.fusion"(%arg157, %arg158, %257, %259, %260) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x4x16xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.subtract %arg247, %314 : tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %317 = stablehlo.multiply %arg249, %316 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %319 = stablehlo.add %317, %318 : tensor<16xf32>
      %320 = stablehlo.rsqrt %319 : tensor<16xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %322 = stablehlo.multiply %315, %321 : tensor<16x256x4x16xf32>
      %323 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<16x256x4x16xf32>
      %325 = stablehlo.add %313, %324 : tensor<16x256x4x16xf32>
      mhlo.return %325 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %262 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%261, %arg159, %arg160) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x512x4x16xf32>, tensor<0xui8>>
    %263 = stablehlo.get_tuple_element %262[0] : (tuple<tensor<16x512x4x16xf32>, tensor<0xui8>>) -> tensor<16x512x4x16xf32>
    %264 = "mhlo.fusion"(%263) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x512x4x16xf32>
      %315 = stablehlo.exponential %314 : tensor<16x512x4x16xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x512x4x16xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x512x4x16xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x512x4x16xf32>
      mhlo.return %318 : tensor<16x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512x4x16xf32>) -> tensor<16x512x4x16xf32>
    %265 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%264, %arg161, %arg162, %257) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>) -> tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>
    %266 = stablehlo.get_tuple_element %265[0] : (tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>) -> tensor<16x256x4x16xf32>
    %267 = "mhlo.fusion"(%266) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>) -> tensor<16xf32>
    %268 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %269 = "mhlo.fusion"(%266, %268) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x4x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x4x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %270 = "mhlo.fusion"(%arg163, %arg164, %266, %268, %269) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x4x16xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.subtract %arg247, %314 : tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %317 = stablehlo.multiply %arg249, %316 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %319 = stablehlo.add %317, %318 : tensor<16xf32>
      %320 = stablehlo.rsqrt %319 : tensor<16xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %322 = stablehlo.multiply %315, %321 : tensor<16x256x4x16xf32>
      %323 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<16x256x4x16xf32>
      %325 = stablehlo.add %313, %324 : tensor<16x256x4x16xf32>
      mhlo.return %325 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %271 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%270, %arg165, %arg166) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<16x513x4x16xf32>, tensor<0xui8>>
    %272 = stablehlo.get_tuple_element %271[0] : (tuple<tensor<16x513x4x16xf32>, tensor<0xui8>>) -> tensor<16x513x4x16xf32>
    %273 = "mhlo.fusion"(%272) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x513x4x16xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x256x4x16xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x16xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x16xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x16xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x256x4x16xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x256x4x16xf32>
      mhlo.return %332 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %274 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%273, %arg167, %arg168, %266) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>) -> tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>
    %275 = stablehlo.get_tuple_element %274[0] : (tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>) -> tensor<16x256x4x16xf32>
    %276 = "mhlo.fusion"(%275) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>) -> tensor<16xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %278 = "mhlo.fusion"(%275, %277) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x4x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x4x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %279 = "mhlo.fusion"(%arg169, %arg170, %275, %277, %278) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x4x16xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.subtract %arg247, %314 : tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %317 = stablehlo.multiply %arg249, %316 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %319 = stablehlo.add %317, %318 : tensor<16xf32>
      %320 = stablehlo.rsqrt %319 : tensor<16xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %322 = stablehlo.multiply %315, %321 : tensor<16x256x4x16xf32>
      %323 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<16x256x4x16xf32>
      %325 = stablehlo.add %313, %324 : tensor<16x256x4x16xf32>
      mhlo.return %325 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %280 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%279, %arg171, %arg172) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x512x4x16xf32>, tensor<0xui8>>
    %281 = stablehlo.get_tuple_element %280[0] : (tuple<tensor<16x512x4x16xf32>, tensor<0xui8>>) -> tensor<16x512x4x16xf32>
    %282 = "mhlo.fusion"(%281) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x512x4x16xf32>
      %315 = stablehlo.exponential %314 : tensor<16x512x4x16xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x512x4x16xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x512x4x16xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x512x4x16xf32>
      mhlo.return %318 : tensor<16x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512x4x16xf32>) -> tensor<16x512x4x16xf32>
    %283 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%282, %arg173, %arg174, %275) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>) -> tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>
    %284 = stablehlo.get_tuple_element %283[0] : (tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>) -> tensor<16x256x4x16xf32>
    %285 = "mhlo.fusion"(%284) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>) -> tensor<16xf32>
    %286 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %287 = "mhlo.fusion"(%284, %286) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x4x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x4x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %288 = "mhlo.fusion"(%arg175, %arg176, %284, %286, %287) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x4x16xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.subtract %arg247, %314 : tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %317 = stablehlo.multiply %arg249, %316 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %319 = stablehlo.add %317, %318 : tensor<16xf32>
      %320 = stablehlo.rsqrt %319 : tensor<16xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %322 = stablehlo.multiply %315, %321 : tensor<16x256x4x16xf32>
      %323 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<16x256x4x16xf32>
      %325 = stablehlo.add %313, %324 : tensor<16x256x4x16xf32>
      mhlo.return %325 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %289 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%288, %arg177, %arg178) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<16x513x4x16xf32>, tensor<0xui8>>
    %290 = stablehlo.get_tuple_element %289[0] : (tuple<tensor<16x513x4x16xf32>, tensor<0xui8>>) -> tensor<16x513x4x16xf32>
    %291 = "mhlo.fusion"(%290) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x513x4x16xf32>):
      %313 = stablehlo.slice %arg245 [0:16, 257:513, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.maximum %313, %314 : tensor<16x256x4x16xf32>
      %316 = stablehlo.slice %arg245 [0:16, 1:257, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
      %317 = stablehlo.slice %arg245 [0:16, 0:1, 0:4, 0:16] : (tensor<16x513x4x16xf32>) -> tensor<16x1x4x16xf32>
      %318 = mhlo.bitcast %317 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %319 = stablehlo.reduce(%318 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
      %320 = stablehlo.broadcast_in_dim %319, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %321 = stablehlo.subtract %317, %320 : tensor<16x1x4x16xf32>
      %322 = stablehlo.exponential %321 : tensor<16x1x4x16xf32>
      %323 = mhlo.bitcast %322 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %324 = stablehlo.reduce(%323 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<16x4x16xf32>, tensor<f32>) -> tensor<16x4xf32>
      %325 = stablehlo.broadcast_in_dim %324, dims = [0, 2] : (tensor<16x4xf32>) -> tensor<16x1x4x16xf32>
      %326 = stablehlo.divide %322, %325 : tensor<16x1x4x16xf32>
      %327 = mhlo.bitcast %326 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x1x4x16xf32>) -> tensor<16x4x16xf32>
      %328 = stablehlo.broadcast_in_dim %327, dims = [0, 2, 3] : (tensor<16x4x16xf32>) -> tensor<16x256x4x16xf32>
      %329 = stablehlo.multiply %316, %328 : tensor<16x256x4x16xf32>
      %330 = stablehlo.reduce(%329 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<16x256x4x16xf32>, tensor<f32>) -> tensor<16x256x4xf32>
      %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1, 2] : (tensor<16x256x4xf32>) -> tensor<16x256x4x16xf32>
      %332 = stablehlo.multiply %315, %331 : tensor<16x256x4x16xf32>
      mhlo.return %332 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x513x4x16xf32>) -> tensor<16x256x4x16xf32>
    %292 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%291, %arg179, %arg180, %284) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>) -> tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>
    %293 = stablehlo.get_tuple_element %292[0] : (tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>) -> tensor<16x256x4x16xf32>
    %294 = "mhlo.fusion"(%293) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>) -> tensor<16xf32>
    %295 = "mhlo.fusion"(%294) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %296 = "mhlo.fusion"(%293, %295) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x4x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x4x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %297 = "mhlo.fusion"(%arg181, %arg182, %293, %295, %296) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x4x16xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.subtract %arg247, %314 : tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %317 = stablehlo.multiply %arg249, %316 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %319 = stablehlo.add %317, %318 : tensor<16xf32>
      %320 = stablehlo.rsqrt %319 : tensor<16xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %322 = stablehlo.multiply %315, %321 : tensor<16x256x4x16xf32>
      %323 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<16x256x4x16xf32>
      %325 = stablehlo.add %313, %324 : tensor<16x256x4x16xf32>
      mhlo.return %325 : tensor<16x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x256x4x16xf32>
    %298 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%297, %arg183, %arg184) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<16x512x4x16xf32>, tensor<0xui8>>
    %299 = stablehlo.get_tuple_element %298[0] : (tuple<tensor<16x512x4x16xf32>, tensor<0xui8>>) -> tensor<16x512x4x16xf32>
    %300 = "mhlo.fusion"(%299) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512x4x16xf32>
      %314 = stablehlo.negate %arg245 : tensor<16x512x4x16xf32>
      %315 = stablehlo.exponential %314 : tensor<16x512x4x16xf32>
      %316 = stablehlo.add %313, %315 : tensor<16x512x4x16xf32>
      %317 = stablehlo.divide %313, %316 : tensor<16x512x4x16xf32>
      %318 = stablehlo.multiply %arg245, %317 : tensor<16x512x4x16xf32>
      mhlo.return %318 : tensor<16x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512x4x16xf32>) -> tensor<16x512x4x16xf32>
    %301 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%300, %arg185, %arg186, %293) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<16x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>) -> tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>
    %302 = stablehlo.get_tuple_element %301[0] : (tuple<tensor<16x256x4x16xf32>, tensor<0xui8>>) -> tensor<16x256x4x16xf32>
    %303 = "mhlo.fusion"(%302) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>):
      %313 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %314 = stablehlo.reduce(%313 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>) -> tensor<16xf32>
    %304 = "mhlo.fusion"(%303) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16xf32>
      mhlo.return %314 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16xf32>) -> tensor<16xf32>
    %305 = "mhlo.fusion"(%302, %304) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x4x16xf32>, %arg246: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.subtract %arg245, %313 : tensor<16x256x4x16xf32>
      %315 = stablehlo.multiply %314, %314 : tensor<16x256x4x16xf32>
      %316 = mhlo.bitcast %315 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.reduce(%316 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x16384xf32>, tensor<f32>) -> tensor<16xf32>
      mhlo.return %317 : tensor<16xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x4x16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %306 = "mhlo.fusion"(%arg187, %arg188, %302, %304, %305) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<16x256x4x16xf32>, %arg248: tensor<16xf32>, %arg249: tensor<16xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %arg248, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %315 = stablehlo.subtract %arg247, %314 : tensor<16x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %316 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %317 = stablehlo.multiply %arg249, %316 : tensor<16xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %318 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
      %319 = stablehlo.add %317, %318 : tensor<16xf32>
      %320 = stablehlo.rsqrt %319 : tensor<16xf32>
      %321 = stablehlo.broadcast_in_dim %320, dims = [0] : (tensor<16xf32>) -> tensor<16x256x4x16xf32>
      %322 = stablehlo.multiply %315, %321 : tensor<16x256x4x16xf32>
      %323 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<16x256x4x16xf32>
      %324 = stablehlo.multiply %322, %323 : tensor<16x256x4x16xf32>
      %325 = stablehlo.add %313, %324 : tensor<16x256x4x16xf32>
      %326 = mhlo.bitcast %325 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x256x4x16xf32>) -> tensor<16x256x2x2x4x4xf32>
      %327 = stablehlo.transpose %326, dims = [2, 4, 3, 5, 0, 1] : (tensor<16x256x2x2x4x4xf32>) -> tensor<2x4x2x4x16x256xf32>
      mhlo.return %327 : tensor<2x4x2x4x16x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<16x256x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<2x4x2x4x16x256xf32>
    %307 = "mhlo.fusion"(%0, %306) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<16x256x8x8xf32>, %arg246: tensor<2x4x2x4x16x256xf32>):
      %313 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %314 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %315 = stablehlo.multiply %313, %314 : tensor<4xi32>
      %316 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %317 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %318 = stablehlo.add %316, %317 : tensor<2x4xi32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %320 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %321 = stablehlo.compare LT, %319, %320 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %322 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %323 = stablehlo.add %319, %322 : tensor<1x1x2x4xi32>
      %324 = stablehlo.select %321, %323, %319 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %325 = mhlo.bitcast %324 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %327 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %328 = stablehlo.compare LT, %318, %327 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %329 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %330 = stablehlo.add %318, %329 : tensor<2x4xi32>
      %331 = stablehlo.select %328, %330, %318 : tensor<2x4xi1>, tensor<2x4xi32>
      %332 = stablehlo.broadcast_in_dim %331, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %333 = stablehlo.concatenate %326, %332, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %334 = mhlo.bitcast %333 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %335 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x16x256xf32>) -> tensor<64x16x256x1x1xf32>
      %336 = "stablehlo.scatter"(%arg245, %334, %335) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %337 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %337 : tensor<f32>
      }) : (tensor<16x256x8x8xf32>, tensor<64x2xi32>, tensor<64x16x256x1x1xf32>) -> tensor<16x256x8x8xf32>
      mhlo.return %336 : tensor<16x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<16x256x8x8xf32>, tensor<2x4x2x4x16x256xf32>) -> tensor<16x256x8x8xf32>
    %308 = stablehlo.custom_call @__cudnn$convForward(%307, %arg189) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[16,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<16x256x8x8xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<16x512x8x8xf32>, tensor<0xui8>>
    %309 = stablehlo.get_tuple_element %308[0] : (tuple<tensor<16x512x8x8xf32>, tensor<0xui8>>) -> tensor<16x512x8x8xf32>
    %310 = "mhlo.fusion"(%arg191, %arg190, %309, %arg242, %arg243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<16x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %313 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %314 = stablehlo.subtract %arg247, %313 : tensor<16x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %315 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %316 = stablehlo.add %arg249, %315 : tensor<512xf32>
      %317 = stablehlo.rsqrt %316 : tensor<512xf32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %319 = stablehlo.multiply %314, %318 : tensor<16x512x8x8xf32>
      %320 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %321 = stablehlo.multiply %319, %320 : tensor<16x512x8x8xf32>
      %322 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<16x512x8x8xf32>
      %323 = stablehlo.add %321, %322 : tensor<16x512x8x8xf32>
      %324 = mhlo.bitcast %323 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<16x512x8x8xf32>) -> tensor<16x512x64xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %325 = stablehlo.reduce(%324 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<16x512x64xf32>, tensor<f32>) -> tensor<16x512xf32>
      mhlo.return %325 : tensor<16x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<16x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<16x512xf32>
    %311 = "mhlo.fusion"(%310, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<16x512xf32>, %arg246: tensor<2x512xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %313 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
      %314 = stablehlo.multiply %arg245, %313 : tensor<16x512xf32>
      %315 = stablehlo.dot_general %314, %arg246, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<16x512xf32>, tensor<2x512xf32>) -> tensor<16x2xf32>
      mhlo.return %315 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x512xf32>, tensor<2x512xf32>) -> tensor<16x2xf32>
    %312 = "mhlo.fusion"(%311, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<16x2xf32>, %arg246: tensor<2xf32>):
      %313 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<2xf32>) -> tensor<16x2xf32>
      %314 = stablehlo.add %313, %arg245 : tensor<16x2xf32>
      mhlo.return %314 : tensor<16x2xf32>
    }) {output_operand_aliasing = []} : (tensor<16x2xf32>, tensor<2xf32>) -> tensor<16x2xf32>
    return %312 : tensor<16x2xf32>
  }
}
