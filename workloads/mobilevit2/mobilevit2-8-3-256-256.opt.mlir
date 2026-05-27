module @jit_func attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func private @gemm_fusion_dot_general.1_computation(%arg0: tensor<8x512xf32>, %arg1: tensor<2x512xf32>) -> tensor<8x2xf32> {
    %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<8x512xf32>
    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x512xf32>, tensor<2x512xf32>) -> tensor<8x2xf32>
    return %2 : tensor<8x2xf32>
  }
  func.func private @region_1.1.clone.54(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.1(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.53(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.2(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.31(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.3(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.52(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.4(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.30(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.5(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.51(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.6(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.29(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.7(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.50(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.8(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.28(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.9(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.49(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.10(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.27(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.11(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.48(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.12(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.26(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.13(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.47(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.14(%arg0: tensor<8x192x4x64xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x4x64xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.25(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.15(%arg0: tensor<8x192x4x64xf32>) -> tensor<8x128xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %1 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.46(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.16(%arg0: tensor<8x192x2x2x8x8xf32>, %arg1: tensor<8xf32>) -> tensor<8x128xf32> {
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x192x2x2x8x8xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x192x2x2x8x8xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x192x2x2x8x8xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x192x2x2x8x8xf32>) -> tensor<8x128x384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
    return %6 : tensor<8x128xf32>
  }
  func.func private @region_1.1.clone.44(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.17(%arg0: tensor<8x128x4x256xf32>, %arg1: tensor<8xf32>) -> tensor<8x256xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x128x4x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %6 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.24(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.18(%arg0: tensor<8x128x4x256xf32>) -> tensor<8x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %1 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.43(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.19(%arg0: tensor<8x128x4x256xf32>, %arg1: tensor<8xf32>) -> tensor<8x256xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x128x4x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %6 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.23(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.20(%arg0: tensor<8x128x4x256xf32>) -> tensor<8x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %1 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.42(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.21(%arg0: tensor<8x128x4x256xf32>, %arg1: tensor<8xf32>) -> tensor<8x256xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x128x4x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %6 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.22(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.22(%arg0: tensor<8x128x4x256xf32>) -> tensor<8x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %1 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.41(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.23(%arg0: tensor<8x128x4x256xf32>, %arg1: tensor<8xf32>) -> tensor<8x256xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x128x4x256xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x128x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %6 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.21(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.24(%arg0: tensor<8x128x4x256xf32>) -> tensor<8x256xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %1 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.40(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.25(%arg0: tensor<8x128x2x2x16x16xf32>, %arg1: tensor<8xf32>) -> tensor<8x256xf32> {
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x128x2x2x16x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x128x2x2x16x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x128x2x2x16x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x128x2x2x16x16xf32>) -> tensor<8x256x512xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
    return %6 : tensor<8x256xf32>
  }
  func.func private @region_1.1.clone.17(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.26(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x8x8xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x512x8x8xf32>) -> tensor<8x512x64xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.reduce(%11 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x512x64xf32>, tensor<f32>) -> tensor<8x512xf32>
    return %12 : tensor<8x512xf32>
  }
  func.func private @region_1.1.clone.62(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x4x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x4x16xf32>
  }
  func.func private @region_1.1.clone.38(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.27(%arg0: tensor<8x256x4x16xf32>) -> tensor<8xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
  func.func private @region_1.1.clone.61(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.1(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x4x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x4x16xf32>
  }
  func.func private @region_1.1.clone.37(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.28(%arg0: tensor<8x256x4x16xf32>) -> tensor<8xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
  func.func private @region_1.1.clone.60(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.2(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x4x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x4x16xf32>
  }
  func.func private @region_1.1.clone.36(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.29(%arg0: tensor<8x256x4x16xf32>) -> tensor<8xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
  func.func private @region_1.1.clone.59(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.3(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x4x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x4x16xf32>
  }
  func.func private @region_1.1.clone.35(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.30(%arg0: tensor<8x256x4x16xf32>) -> tensor<8xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
  func.func private @region_1.1.clone.58(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.4(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x4x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x4x16xf32>
  }
  func.func private @region_1.1.clone.34(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.31(%arg0: tensor<8x256x4x16xf32>) -> tensor<8xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
  func.func private @region_1.1.clone.57(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.5(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x4x16xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x4x16xf32>
  }
  func.func private @region_1.1.clone.33(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.32(%arg0: tensor<8x256x4x16xf32>) -> tensor<8xf32> {
    %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
  func.func private @region_1.1.clone.56(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce_subtract.6(%arg0: tensor<8x256x2x2x4x4xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x2x2x4x4xf32>) {
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.multiply %arg1, %0 : tensor<8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8x256x2x2x4x4xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<8x256x2x2x4x4xf32>
    %4 = stablehlo.multiply %3, %3 : tensor<8x256x2x2x4x4xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x256x2x2x4x4xf32>) -> tensor<8x16384xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
    return %6, %3 : tensor<8xf32>, tensor<8x256x2x2x4x4xf32>
  }
  func.func private @region_1.1.clone.55(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.33(%arg0: tensor<8x256x8x8xf32>) -> tensor<64x8xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x8x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x8x256x1x1xf32>) -> tensor<64x8x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x8x256xf32>, tensor<f32>) -> tensor<64x8xf32>
    return %24 : tensor<64x8xf32>
  }
  func.func private @region_1.1.clone.45(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.34(%arg0: tensor<8x192x16x16xf32>) -> tensor<256x8xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 192, 1, 1>}> : (tensor<8x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x8x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x8x192x1x1xf32>) -> tensor<256x8x192xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x8x192xf32>, tensor<f32>) -> tensor<256x8xf32>
    return %24 : tensor<256x8xf32>
  }
  func.func private @region_1.1.clone.39(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_reduce.35(%arg0: tensor<8x128x32x32xf32>) -> tensor<1024x8xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 128, 1, 1>}> : (tensor<8x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x8x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x8x128x1x1xf32>) -> tensor<1024x8x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.reduce(%20 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x8x128xf32>, tensor<f32>) -> tensor<1024x8xf32>
    return %21 : tensor<1024x8xf32>
  }
  func.func private @region_1.1.clone.20(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter(%arg0: tensor<8x256x8x8xf32>, %arg1: tensor<2x4x2x4x8x256xf32>) -> tensor<8x256x8x8xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x8x256xf32>) -> tensor<64x8x256x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<8x256x8x8xf32>, tensor<64x2xi32>, tensor<64x8x256x1x1xf32>) -> tensor<8x256x8x8xf32>
    return %23 : tensor<8x256x8x8xf32>
  }
  func.func private @fused_transpose(%arg0: tensor<256xf32>, %arg1: tensor<8x256x4x16xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<2x4x2x4x8x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %4 = stablehlo.add %2, %3 : tensor<8xf32>
    %5 = stablehlo.rsqrt %4 : tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %7 = stablehlo.multiply %arg1, %6 : tensor<8x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<8x256x4x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x256x2x2x4x4xf32>
    %12 = stablehlo.transpose %11, dims = [2, 4, 3, 5, 0, 1] : (tensor<8x256x2x2x4x4xf32>) -> tensor<2x4x2x4x8x256xf32>
    return %12 : tensor<2x4x2x4x8x256xf32>
  }
  func.func private @fused_transpose.1(%arg0: tensor<8x256x8x8xf32>) -> tensor<8x256x2x2x4x4xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x8x256x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x8x256x1x1xf32>) -> tensor<2x4x2x4x8x256xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x8x256xf32>) -> tensor<8x256x2x2x4x4xf32>
    return %24 : tensor<8x256x2x2x4x4xf32>
  }
  func.func private @region_1.1.clone.19(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.1(%arg0: tensor<8x192x16x16xf32>, %arg1: tensor<2x8x2x8x8x192xf32>) -> tensor<8x192x16x16xf32> {
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
    %22 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x8x192xf32>) -> tensor<256x8x192x1x1xf32>
    %23 = "stablehlo.scatter"(%arg0, %21, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %24 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %24 : tensor<f32>
    }) : (tensor<8x192x16x16xf32>, tensor<256x2xi32>, tensor<256x8x192x1x1xf32>) -> tensor<8x192x16x16xf32>
    return %23 : tensor<8x192x16x16xf32>
  }
  func.func private @fused_transpose.2(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<2x8x2x8x8x192xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x192x2x2x8x8xf32>
    %15 = stablehlo.transpose %14, dims = [2, 4, 3, 5, 0, 1] : (tensor<8x192x2x2x8x8xf32>) -> tensor<2x8x2x8x8x192xf32>
    return %15 : tensor<2x8x2x8x8x192xf32>
  }
  func.func private @fused_transpose.3(%arg0: tensor<8x192x16x16xf32>) -> tensor<8x192x2x2x8x8xf32> {
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
    %22 = "stablehlo.gather"(%arg0, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 192, 1, 1>}> : (tensor<8x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x8x192x1x1xf32>
    %23 = mhlo.bitcast %22 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x8x192x1x1xf32>) -> tensor<2x8x2x8x8x192xf32>
    %24 = stablehlo.transpose %23, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x8x192xf32>) -> tensor<8x192x2x2x8x8xf32>
    return %24 : tensor<8x192x2x2x8x8xf32>
  }
  func.func private @region_1.1.clone.18(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @fused_scatter.2(%arg0: tensor<8x128x32x32xf32>, %arg1: tensor<2x16x2x16x8x128xf32>) -> tensor<8x128x32x32xf32> {
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
    %19 = mhlo.bitcast %arg1 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x8x128xf32>) -> tensor<1024x8x128x1x1xf32>
    %20 = "stablehlo.scatter"(%arg0, %18, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %21 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }) : (tensor<8x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x8x128x1x1xf32>) -> tensor<8x128x32x32xf32>
    return %20 : tensor<8x128x32x32xf32>
  }
  func.func private @fused_transpose.4(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<8x128x4x256xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<2x16x2x16x8x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x128x4x256xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x128x4x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x128x4x256xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x128x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x128x2x2x16x16xf32>
    %15 = stablehlo.transpose %14, dims = [2, 4, 3, 5, 0, 1] : (tensor<8x128x2x2x16x16xf32>) -> tensor<2x16x2x16x8x128xf32>
    return %15 : tensor<2x16x2x16x8x128xf32>
  }
  func.func private @fused_transpose.5(%arg0: tensor<8x128x32x32xf32>) -> tensor<8x128x2x2x16x16xf32> {
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
    %19 = "stablehlo.gather"(%arg0, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 128, 1, 1>}> : (tensor<8x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x8x128x1x1xf32>
    %20 = mhlo.bitcast %19 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x8x128x1x1xf32>) -> tensor<2x16x2x16x8x128xf32>
    %21 = stablehlo.transpose %20, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x8x128xf32>) -> tensor<8x128x2x2x16x16xf32>
    return %21 : tensor<8x128x2x2x16x16xf32>
  }
  func.func private @fused_add(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<8x128x2x2x16x16xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<1x8x131072xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<1x8x131072xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %3 = stablehlo.multiply %arg4, %2 : tensor<8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<8xf32>) -> tensor<8x128x2x2x16x16xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<8x128x2x2x16x16xf32>
    %6 = stablehlo.multiply %arg3, %2 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %8 = stablehlo.add %6, %7 : tensor<8xf32>
    %9 = stablehlo.rsqrt %8 : tensor<8xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<8xf32>) -> tensor<8x128x2x2x16x16xf32>
    %11 = stablehlo.multiply %5, %10 : tensor<8x128x2x2x16x16xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x128x2x2x16x16xf32>) -> tensor<1x8x131072xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<1x8x131072xf32>
    %15 = stablehlo.multiply %12, %14 : tensor<1x8x131072xf32>
    %16 = stablehlo.add %1, %15 : tensor<1x8x131072xf32>
    return %16 : tensor<1x8x131072xf32>
  }
  func.func private @fused_add.1(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x2x2x8x8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<1x8x49152xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<1x8x49152xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %3 = stablehlo.multiply %arg4, %2 : tensor<8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<8xf32>) -> tensor<8x192x2x2x8x8xf32>
    %5 = stablehlo.subtract %arg2, %4 : tensor<8x192x2x2x8x8xf32>
    %6 = stablehlo.multiply %arg3, %2 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %8 = stablehlo.add %6, %7 : tensor<8xf32>
    %9 = stablehlo.rsqrt %8 : tensor<8xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<8xf32>) -> tensor<8x192x2x2x8x8xf32>
    %11 = stablehlo.multiply %5, %10 : tensor<8x192x2x2x8x8xf32>
    %12 = mhlo.bitcast %11 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x192x2x2x8x8xf32>) -> tensor<1x8x49152xf32>
    %13 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<1x8x49152xf32>
    %15 = stablehlo.multiply %12, %14 : tensor<1x8x49152xf32>
    %16 = stablehlo.add %1, %15 : tensor<1x8x49152xf32>
    return %16 : tensor<1x8x49152xf32>
  }
  func.func private @fused_add.2(%arg0: tensor<8x256x2x2x4x4xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<1x8x16384xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %1 = mhlo.bitcast %0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<1x8x16384xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %3 = stablehlo.multiply %arg3, %2 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %5 = stablehlo.add %3, %4 : tensor<8xf32>
    %6 = stablehlo.rsqrt %5 : tensor<8xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<8xf32>) -> tensor<8x256x2x2x4x4xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<8x256x2x2x4x4xf32>
    %9 = mhlo.bitcast %8 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x256x2x2x4x4xf32>) -> tensor<1x8x16384xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %11 = mhlo.bitcast %10 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<1x8x16384xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<1x8x16384xf32>
    %13 = stablehlo.add %1, %12 : tensor<1x8x16384xf32>
    return %13 : tensor<1x8x16384xf32>
  }
  func.func private @fused_multiply(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<8x128x128x128xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x128x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x128x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x128x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x128x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<8x128x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<8x128x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x128x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x128x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x128x128x128xf32>
    return %13 : tensor<8x128x128x128xf32>
  }
  func.func private @fused_add.3(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<8x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x64x128x128xf32>
    return %7 : tensor<8x64x128x128xf32>
  }
  func.func private @fused_multiply.1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<8x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<8x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<8x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x64x128x128xf32>
    return %13 : tensor<8x64x128x128xf32>
  }
  func.func private @fused_multiply.2(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<8x64x128x128xf32>, %arg4: tensor<64xf32>) -> tensor<8x64x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x64x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x64x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x64x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x64x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x64x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<8x64x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<8x64x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x64x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x64x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x64x128x128xf32>
    return %13 : tensor<8x64x128x128xf32>
  }
  func.func private @fused_multiply.3(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<8x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<8x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x256x64x64xf32>
    return %13 : tensor<8x256x64x64xf32>
  }
  func.func private @fused_multiply.4(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<8x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<8x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x256x64x64xf32>
    return %13 : tensor<8x256x64x64xf32>
  }
  func.func private @fused_multiply.5(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8x256x64x64xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x256x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x256x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x256x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x256x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<8x256x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<8x256x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x256x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x256x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x256x64x64xf32>
    return %13 : tensor<8x256x64x64xf32>
  }
  func.func private @fused_add.4(%arg0: tensor<8x128x64x64xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<8x128x64x64xf32>, %arg5: tensor<128xf32>) -> tensor<8x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %1 = stablehlo.subtract %arg4, %0 : tensor<8x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x128x64x64xf32>
    %8 = stablehlo.add %arg0, %7 : tensor<8x128x64x64xf32>
    return %8 : tensor<8x128x64x64xf32>
  }
  func.func private @fused_add.5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<8x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x128x64x64xf32>
    return %7 : tensor<8x128x64x64xf32>
  }
  func.func private @fused_multiply.6(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<8x32x128x128xf32>, %arg4: tensor<32xf32>) -> tensor<8x32x128x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x32x128x128xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x32x128x128xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x32x128x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x32x128x128xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x32x128x128xf32>
    %9 = stablehlo.negate %7 : tensor<8x32x128x128xf32>
    %10 = stablehlo.exponential %9 : tensor<8x32x128x128xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x32x128x128xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x32x128x128xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x32x128x128xf32>
    return %13 : tensor<8x32x128x128xf32>
  }
  func.func private @fused_multiply.7(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<8x128x64x64xf32>, %arg4: tensor<128xf32>) -> tensor<8x128x64x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x128x64x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x128x64x64xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x128x64x64xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x128x64x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x64x64xf32>
    %9 = stablehlo.negate %7 : tensor<8x128x64x64xf32>
    %10 = stablehlo.exponential %9 : tensor<8x128x64x64xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x128x64x64xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x128x64x64xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x128x64x64xf32>
    return %13 : tensor<8x128x64x64xf32>
  }
  func.func private @fused_multiply.8(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<8x512x32x32xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
    %1 = stablehlo.subtract %arg3, %0 : tensor<8x512x32x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8x512x32x32xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<8x512x32x32xf32>
    %6 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
    %7 = stablehlo.add %5, %6 : tensor<8x512x32x32xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x32x32xf32>
    %9 = stablehlo.negate %7 : tensor<8x512x32x32xf32>
    %10 = stablehlo.exponential %9 : tensor<8x512x32x32xf32>
    %11 = stablehlo.add %8, %10 : tensor<8x512x32x32xf32>
    %12 = stablehlo.divide %8, %11 : tensor<8x512x32x32xf32>
    %13 = stablehlo.multiply %7, %12 : tensor<8x512x32x32xf32>
    return %13 : tensor<8x512x32x32xf32>
  }
  func.func private @fused_add.6(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x32x32xf32>
    return %10 : tensor<8x256x32x32xf32>
  }
  func.func private @fused_add.7(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x32x32xf32>
    return %10 : tensor<8x256x32x32xf32>
  }
  func.func private @fused_multiply.9(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x32x32xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x32x32xf32>
    %12 = stablehlo.negate %10 : tensor<8x256x32x32xf32>
    %13 = stablehlo.exponential %12 : tensor<8x256x32x32xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x256x32x32xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x256x32x32xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x256x32x32xf32>
    return %16 : tensor<8x256x32x32xf32>
  }
  func.func private @fused_multiply.10(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<8x256x32x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<8x256x32x32xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x256x32x32xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<256xf32>
    %4 = stablehlo.rsqrt %3 : tensor<256xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x256x32x32xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x256x32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x256x32x32xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x32x32xf32>
    %12 = stablehlo.negate %10 : tensor<8x256x32x32xf32>
    %13 = stablehlo.exponential %12 : tensor<8x256x32x32xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x256x32x32xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x256x32x32xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x256x32x32xf32>
    return %16 : tensor<8x256x32x32xf32>
  }
  func.func private @fused_multiply.11(%arg0: tensor<8x256x4x256xf32>) -> tensor<8x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<8x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x256x4x256xf32>
    return %5 : tensor<8x256x4x256xf32>
  }
  func.func private @fused_multiply.12(%arg0: tensor<8x256x4x256xf32>) -> tensor<8x256x4x256xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x256xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x256x4x256xf32>
    %2 = stablehlo.exponential %1 : tensor<8x256x4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x256x4x256xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x256x4x256xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x256x4x256xf32>
    return %5 : tensor<8x256x4x256xf32>
  }
  func.func private @fused_multiply.13(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<8x768x16x16xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<8x768x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x768x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x768x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x768x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x768x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<8x768x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<8x768x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x768x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x768x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x768x16x16xf32>
    return %16 : tensor<8x768x16x16xf32>
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
  func.func private @fused_computation.193(%arg0: tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 129:257, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:129, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x256xf32>) -> tensor<8x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x128x4x256xf32>, tensor<f32>) -> tensor<8x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x128x4xf32>) -> tensor<8x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x128x4x256xf32>
    return %19 : tensor<8x128x4x256xf32>
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
  func.func private @fused_computation.194(%arg0: tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 129:257, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x4x256xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x128x4x256xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:129, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x1x4x256xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x256xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x256xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x256xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x256xf32>) -> tensor<8x128x4x256xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x128x4x256xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x128x4x256xf32>, tensor<f32>) -> tensor<8x128x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x128x4xf32>) -> tensor<8x128x4x256xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x128x4x256xf32>
    return %19 : tensor<8x128x4x256xf32>
  }
  func.func private @fused_add.8(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<8x128x4x256xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x128x4x256xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x128x4x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x128x4x256xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x128x4x256xf32>
    return %13 : tensor<8x128x4x256xf32>
  }
  func.func private @fused_add.9(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<8x128x4x256xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x128x4x256xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x128x4x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x128x4x256xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x128x4x256xf32>
    return %13 : tensor<8x128x4x256xf32>
  }
  func.func private @fused_add.10(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<8x128x4x256xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x128x4x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x128x4x256xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x128x4x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x128x4x256xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x128x4x256xf32>
    return %13 : tensor<8x128x4x256xf32>
  }
  func.func private @fused_multiply.14(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x16x16xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<8x512x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<8x512x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x512x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x512x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x512x16x16xf32>
    return %16 : tensor<8x512x16x16xf32>
  }
  func.func private @fused_multiply.15(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<8x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<8x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x384x16x16xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x384x16x16xf32>
    %12 = stablehlo.negate %10 : tensor<8x384x16x16xf32>
    %13 = stablehlo.exponential %12 : tensor<8x384x16x16xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x384x16x16xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x384x16x16xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x384x16x16xf32>
    return %16 : tensor<8x384x16x16xf32>
  }
  func.func private @fused_multiply.16(%arg0: tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<8x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x384x4x64xf32>
    return %5 : tensor<8x384x4x64xf32>
  }
  func.func private @fused_multiply.17(%arg0: tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<8x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x384x4x64xf32>
    return %5 : tensor<8x384x4x64xf32>
  }
  func.func private @fused_multiply.18(%arg0: tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<8x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x384x4x64xf32>
    return %5 : tensor<8x384x4x64xf32>
  }
  func.func private @fused_multiply.19(%arg0: tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x384x4x64xf32>
    %2 = stablehlo.exponential %1 : tensor<8x384x4x64xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x384x4x64xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x384x4x64xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x384x4x64xf32>
    return %5 : tensor<8x384x4x64xf32>
  }
  func.func private @fused_add.11(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<8x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<8x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x384x16x16xf32>
    return %10 : tensor<8x384x16x16xf32>
  }
  func.func private @fused_add.12(%arg0: tensor<384xf32>, %arg1: tensor<384xf32>, %arg2: tensor<8x384x16x16xf32>, %arg3: tensor<384xf32>, %arg4: tensor<384xf32>) -> tensor<8x384x16x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x384x16x16xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<384xf32>
    %4 = stablehlo.rsqrt %3 : tensor<384xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x384x16x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x384x16x16xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x384x16x16xf32>
    return %10 : tensor<8x384x16x16xf32>
  }
  func.func private @fused_multiply.20(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<8x768x8x8xf32>, %arg3: tensor<768xf32>, %arg4: tensor<768xf32>) -> tensor<8x768x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x768x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<768xf32>
    %4 = stablehlo.rsqrt %3 : tensor<768xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x768x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x768x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x768x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<8x768x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<8x768x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x768x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x768x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x768x8x8xf32>
    return %16 : tensor<8x768x8x8xf32>
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
  func.func private @fused_computation.252(%arg0: tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x192x4x64xf32>
    return %19 : tensor<8x192x4x64xf32>
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
  func.func private @fused_computation.253(%arg0: tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x192x4x64xf32>
    return %19 : tensor<8x192x4x64xf32>
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
  func.func private @fused_computation.254(%arg0: tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x192x4x64xf32>
    return %19 : tensor<8x192x4x64xf32>
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
  func.func private @fused_computation.255(%arg0: tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x192x4x64xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x64xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x64xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x64xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x192x4x64xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x192x4x64xf32>
    return %19 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.13(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.14(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.15(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.16(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.17(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.18(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_add.19(%arg0: tensor<192xf32>, %arg1: tensor<192xf32>, %arg2: tensor<8x192x4x64xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x192x4x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg4, %1 : tensor<8xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %4 = stablehlo.subtract %arg2, %3 : tensor<8x192x4x64xf32>
    %5 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %7 = stablehlo.add %5, %6 : tensor<8xf32>
    %8 = stablehlo.rsqrt %7 : tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %10 = stablehlo.multiply %4, %9 : tensor<8x192x4x64xf32>
    %11 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<8x192x4x64xf32>
    %13 = stablehlo.add %0, %12 : tensor<8x192x4x64xf32>
    return %13 : tensor<8x192x4x64xf32>
  }
  func.func private @fused_multiply.21(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x8x8xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x8x8xf32>
    %12 = stablehlo.negate %10 : tensor<8x512x8x8xf32>
    %13 = stablehlo.exponential %12 : tensor<8x512x8x8xf32>
    %14 = stablehlo.add %11, %13 : tensor<8x512x8x8xf32>
    %15 = stablehlo.divide %11, %14 : tensor<8x512x8x8xf32>
    %16 = stablehlo.multiply %10, %15 : tensor<8x512x8x8xf32>
    return %16 : tensor<8x512x8x8xf32>
  }
  func.func private @fused_multiply.22(%arg0: tensor<8x512x4x16xf32>) -> tensor<8x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<8x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x512x4x16xf32>
    return %5 : tensor<8x512x4x16xf32>
  }
  func.func private @fused_multiply.23(%arg0: tensor<8x512x4x16xf32>) -> tensor<8x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<8x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x512x4x16xf32>
    return %5 : tensor<8x512x4x16xf32>
  }
  func.func private @fused_multiply.24(%arg0: tensor<8x512x4x16xf32>) -> tensor<8x512x4x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x4x16xf32>
    %1 = stablehlo.negate %arg0 : tensor<8x512x4x16xf32>
    %2 = stablehlo.exponential %1 : tensor<8x512x4x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x512x4x16xf32>
    %4 = stablehlo.divide %0, %3 : tensor<8x512x4x16xf32>
    %5 = stablehlo.multiply %arg0, %4 : tensor<8x512x4x16xf32>
    return %5 : tensor<8x512x4x16xf32>
  }
  func.func private @fused_add.20(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<8x512x8x8xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tensor<8x512x8x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %1 = stablehlo.subtract %arg2, %0 : tensor<8x512x8x8xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %3 = stablehlo.add %arg4, %2 : tensor<512xf32>
    %4 = stablehlo.rsqrt %3 : tensor<512xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %6 = stablehlo.multiply %1, %5 : tensor<8x512x8x8xf32>
    %7 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<8x512x8x8xf32>
    %9 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %10 = stablehlo.add %8, %9 : tensor<8x512x8x8xf32>
    return %10 : tensor<8x512x8x8xf32>
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
  func.func private @fused_computation.313(%arg0: tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 257:513, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:257, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x16xf32>) -> tensor<8x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x256x4x16xf32>, tensor<f32>) -> tensor<8x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x256x4xf32>) -> tensor<8x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x256x4x16xf32>
    return %19 : tensor<8x256x4x16xf32>
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
  func.func private @fused_computation.314(%arg0: tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 257:513, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:257, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x16xf32>) -> tensor<8x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x256x4x16xf32>, tensor<f32>) -> tensor<8x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x256x4xf32>) -> tensor<8x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x256x4x16xf32>
    return %19 : tensor<8x256x4x16xf32>
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
  func.func private @fused_computation.315(%arg0: tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.slice %arg0 [0:8, 257:513, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x16xf32>
    %2 = stablehlo.maximum %0, %1 : tensor<8x256x4x16xf32>
    %3 = stablehlo.slice %arg0 [0:8, 1:257, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %4 = stablehlo.slice %arg0 [0:8, 0:1, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x1x4x16xf32>
    %5 = mhlo.bitcast %4 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
    %8 = stablehlo.subtract %4, %7 : tensor<8x1x4x16xf32>
    %9 = stablehlo.exponential %8 : tensor<8x1x4x16xf32>
    %10 = mhlo.bitcast %9 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
    %13 = stablehlo.divide %9, %12 : tensor<8x1x4x16xf32>
    %14 = mhlo.bitcast %13 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 2, 3] : (tensor<8x4x16xf32>) -> tensor<8x256x4x16xf32>
    %16 = stablehlo.multiply %3, %15 : tensor<8x256x4x16xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x256x4x16xf32>, tensor<f32>) -> tensor<8x256x4xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<8x256x4xf32>) -> tensor<8x256x4x16xf32>
    %19 = stablehlo.multiply %2, %18 : tensor<8x256x4x16xf32>
    return %19 : tensor<8x256x4x16xf32>
  }
  func.func private @fused_add.21(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %4 = stablehlo.add %2, %3 : tensor<8xf32>
    %5 = stablehlo.rsqrt %4 : tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<8x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<8x256x4x16xf32>
    return %10 : tensor<8x256x4x16xf32>
  }
  func.func private @fused_add.22(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %4 = stablehlo.add %2, %3 : tensor<8xf32>
    %5 = stablehlo.rsqrt %4 : tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<8x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<8x256x4x16xf32>
    return %10 : tensor<8x256x4x16xf32>
  }
  func.func private @fused_add.23(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %4 = stablehlo.add %2, %3 : tensor<8xf32>
    %5 = stablehlo.rsqrt %4 : tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<8x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<8x256x4x16xf32>
    return %10 : tensor<8x256x4x16xf32>
  }
  func.func private @fused_add.24(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %4 = stablehlo.add %2, %3 : tensor<8xf32>
    %5 = stablehlo.rsqrt %4 : tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<8x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<8x256x4x16xf32>
    return %10 : tensor<8x256x4x16xf32>
  }
  func.func private @fused_add.25(%arg0: tensor<8x256x4x16xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<8xf32>) -> tensor<8x256x4x16xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %2 = stablehlo.multiply %arg3, %1 : tensor<8xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %4 = stablehlo.add %2, %3 : tensor<8xf32>
    %5 = stablehlo.rsqrt %4 : tensor<8xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %7 = stablehlo.multiply %arg0, %6 : tensor<8x256x4x16xf32>
    %8 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<8x256x4x16xf32>
    %10 = stablehlo.add %0, %9 : tensor<8x256x4x16xf32>
    return %10 : tensor<8x256x4x16xf32>
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
  func.func private @fused_rsqrt.4(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.5(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.6(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.7(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<128xf32>
    %2 = stablehlo.rsqrt %1 : tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  func.func private @fused_rsqrt.8(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.9(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.10(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<64xf32>
    %2 = stablehlo.rsqrt %1 : tensor<64xf32>
    return %2 : tensor<64xf32>
  }
  func.func private @fused_rsqrt.11(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<32xf32>
    %2 = stablehlo.rsqrt %1 : tensor<32xf32>
    return %2 : tensor<32xf32>
  }
  func.func private @fused_add.26(%arg0: tensor<8x2xf32>, %arg1: tensor<2xf32>) -> tensor<8x2xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<8x2xf32>
    %1 = stablehlo.add %0, %arg0 : tensor<8x2xf32>
    return %1 : tensor<8x2xf32>
  }
  func.func private @fused_broadcast() -> tensor<8x192x16x16xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x16x16xf32>
    return %0 : tensor<8x192x16x16xf32>
  }
  func.func private @fused_broadcast.1() -> tensor<8x128x32x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x32x32xf32>
    return %0 : tensor<8x128x32x32xf32>
  }
  func.func private @fused_broadcast.2() -> tensor<8x256x8x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x8x8xf32>
    return %0 : tensor<8x256x8x8xf32>
  }
  func.func private @fused_reduce.36(%arg0: tensor<1024x8xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x8xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.37(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.38(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.39(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.40(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.41(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.42(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.43(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.44(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.45(%arg0: tensor<8x256xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.46(%arg0: tensor<256x8xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x8xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.47(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.48(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.49(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.50(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.51(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.52(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.53(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.54(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.55(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.56(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.57(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.58(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.59(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.60(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.61(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.62(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.63(%arg0: tensor<8x128xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @fused_reduce.64(%arg0: tensor<64x8xf32>) -> tensor<8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x8xf32>, tensor<f32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func @main(%arg0: tensor<32x3x3x3xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<64x32x1x1xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x1x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64x64x1x1xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<128x64x1x1xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x1x3x3xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128x128x1x1xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<256x128x1x1xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256x1x3x3xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<128x256x1x1xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<256x128x1x1xf32>, %arg31: tensor<256xf32>, %arg32: tensor<256xf32>, %arg33: tensor<256x1x3x3xf32>, %arg34: tensor<256xf32>, %arg35: tensor<256xf32>, %arg36: tensor<256x256x1x1xf32>, %arg37: tensor<256xf32>, %arg38: tensor<256xf32>, %arg39: tensor<256x1x3x3xf32>, %arg40: tensor<256xf32>, %arg41: tensor<256xf32>, %arg42: tensor<128x256x1x1xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<257x128x1x1xf32>, %arg46: tensor<257xf32>, %arg47: tensor<128x128x1x1xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x1x1xf32>, %arg52: tensor<256xf32>, %arg53: tensor<128x256x1x1xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<257x128x1x1xf32>, %arg58: tensor<257xf32>, %arg59: tensor<128x128x1x1xf32>, %arg60: tensor<128xf32>, %arg61: tensor<128xf32>, %arg62: tensor<128xf32>, %arg63: tensor<256x128x1x1xf32>, %arg64: tensor<256xf32>, %arg65: tensor<128x256x1x1xf32>, %arg66: tensor<128xf32>, %arg67: tensor<128xf32>, %arg68: tensor<128xf32>, %arg69: tensor<256x128x1x1xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256xf32>, %arg72: tensor<512x256x1x1xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512x1x3x3xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<384x512x1x1xf32>, %arg79: tensor<384xf32>, %arg80: tensor<384xf32>, %arg81: tensor<384x1x3x3xf32>, %arg82: tensor<384xf32>, %arg83: tensor<384xf32>, %arg84: tensor<192x384x1x1xf32>, %arg85: tensor<192xf32>, %arg86: tensor<192xf32>, %arg87: tensor<385x192x1x1xf32>, %arg88: tensor<385xf32>, %arg89: tensor<192x192x1x1xf32>, %arg90: tensor<192xf32>, %arg91: tensor<192xf32>, %arg92: tensor<192xf32>, %arg93: tensor<384x192x1x1xf32>, %arg94: tensor<384xf32>, %arg95: tensor<192x384x1x1xf32>, %arg96: tensor<192xf32>, %arg97: tensor<192xf32>, %arg98: tensor<192xf32>, %arg99: tensor<385x192x1x1xf32>, %arg100: tensor<385xf32>, %arg101: tensor<192x192x1x1xf32>, %arg102: tensor<192xf32>, %arg103: tensor<192xf32>, %arg104: tensor<192xf32>, %arg105: tensor<384x192x1x1xf32>, %arg106: tensor<384xf32>, %arg107: tensor<192x384x1x1xf32>, %arg108: tensor<192xf32>, %arg109: tensor<192xf32>, %arg110: tensor<192xf32>, %arg111: tensor<385x192x1x1xf32>, %arg112: tensor<385xf32>, %arg113: tensor<192x192x1x1xf32>, %arg114: tensor<192xf32>, %arg115: tensor<192xf32>, %arg116: tensor<192xf32>, %arg117: tensor<384x192x1x1xf32>, %arg118: tensor<384xf32>, %arg119: tensor<192x384x1x1xf32>, %arg120: tensor<192xf32>, %arg121: tensor<192xf32>, %arg122: tensor<192xf32>, %arg123: tensor<385x192x1x1xf32>, %arg124: tensor<385xf32>, %arg125: tensor<192x192x1x1xf32>, %arg126: tensor<192xf32>, %arg127: tensor<192xf32>, %arg128: tensor<192xf32>, %arg129: tensor<384x192x1x1xf32>, %arg130: tensor<384xf32>, %arg131: tensor<192x384x1x1xf32>, %arg132: tensor<192xf32>, %arg133: tensor<192xf32>, %arg134: tensor<192xf32>, %arg135: tensor<384x192x1x1xf32>, %arg136: tensor<384xf32>, %arg137: tensor<384xf32>, %arg138: tensor<768x384x1x1xf32>, %arg139: tensor<768xf32>, %arg140: tensor<768xf32>, %arg141: tensor<768x1x3x3xf32>, %arg142: tensor<768xf32>, %arg143: tensor<768xf32>, %arg144: tensor<512x768x1x1xf32>, %arg145: tensor<512xf32>, %arg146: tensor<512xf32>, %arg147: tensor<512x1x3x3xf32>, %arg148: tensor<512xf32>, %arg149: tensor<512xf32>, %arg150: tensor<256x512x1x1xf32>, %arg151: tensor<256xf32>, %arg152: tensor<256xf32>, %arg153: tensor<513x256x1x1xf32>, %arg154: tensor<513xf32>, %arg155: tensor<256x256x1x1xf32>, %arg156: tensor<256xf32>, %arg157: tensor<256xf32>, %arg158: tensor<256xf32>, %arg159: tensor<512x256x1x1xf32>, %arg160: tensor<512xf32>, %arg161: tensor<256x512x1x1xf32>, %arg162: tensor<256xf32>, %arg163: tensor<256xf32>, %arg164: tensor<256xf32>, %arg165: tensor<513x256x1x1xf32>, %arg166: tensor<513xf32>, %arg167: tensor<256x256x1x1xf32>, %arg168: tensor<256xf32>, %arg169: tensor<256xf32>, %arg170: tensor<256xf32>, %arg171: tensor<512x256x1x1xf32>, %arg172: tensor<512xf32>, %arg173: tensor<256x512x1x1xf32>, %arg174: tensor<256xf32>, %arg175: tensor<256xf32>, %arg176: tensor<256xf32>, %arg177: tensor<513x256x1x1xf32>, %arg178: tensor<513xf32>, %arg179: tensor<256x256x1x1xf32>, %arg180: tensor<256xf32>, %arg181: tensor<256xf32>, %arg182: tensor<256xf32>, %arg183: tensor<512x256x1x1xf32>, %arg184: tensor<512xf32>, %arg185: tensor<256x512x1x1xf32>, %arg186: tensor<256xf32>, %arg187: tensor<256xf32>, %arg188: tensor<256xf32>, %arg189: tensor<512x256x1x1xf32>, %arg190: tensor<512xf32>, %arg191: tensor<512xf32>, %arg192: tensor<2x512xf32>, %arg193: tensor<2xf32>, %arg194: tensor<32xf32>, %arg195: tensor<32xf32>, %arg196: tensor<64xf32>, %arg197: tensor<64xf32>, %arg198: tensor<64xf32>, %arg199: tensor<64xf32>, %arg200: tensor<64xf32>, %arg201: tensor<64xf32>, %arg202: tensor<128xf32>, %arg203: tensor<128xf32>, %arg204: tensor<128xf32>, %arg205: tensor<128xf32>, %arg206: tensor<128xf32>, %arg207: tensor<128xf32>, %arg208: tensor<256xf32>, %arg209: tensor<256xf32>, %arg210: tensor<256xf32>, %arg211: tensor<256xf32>, %arg212: tensor<128xf32>, %arg213: tensor<128xf32>, %arg214: tensor<256xf32>, %arg215: tensor<256xf32>, %arg216: tensor<256xf32>, %arg217: tensor<256xf32>, %arg218: tensor<256xf32>, %arg219: tensor<256xf32>, %arg220: tensor<256xf32>, %arg221: tensor<256xf32>, %arg222: tensor<256xf32>, %arg223: tensor<256xf32>, %arg224: tensor<512xf32>, %arg225: tensor<512xf32>, %arg226: tensor<512xf32>, %arg227: tensor<512xf32>, %arg228: tensor<384xf32>, %arg229: tensor<384xf32>, %arg230: tensor<384xf32>, %arg231: tensor<384xf32>, %arg232: tensor<384xf32>, %arg233: tensor<384xf32>, %arg234: tensor<768xf32>, %arg235: tensor<768xf32>, %arg236: tensor<768xf32>, %arg237: tensor<768xf32>, %arg238: tensor<512xf32>, %arg239: tensor<512xf32>, %arg240: tensor<512xf32>, %arg241: tensor<512xf32>, %arg242: tensor<512xf32>, %arg243: tensor<512xf32>, %arg244: tensor<8x3x256x256xf32>) -> tensor<8x2xf32> {
    %0 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x8x8xf32>
      mhlo.return %299 : tensor<8x256x8x8xf32>
    }) {output_operand_aliasing = []} : () -> tensor<8x256x8x8xf32>
    %1 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x16x16xf32>
      mhlo.return %299 : tensor<8x192x16x16xf32>
    }) {output_operand_aliasing = []} : () -> tensor<8x192x16x16xf32>
    %2 = "mhlo.fusion"(%arg225) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<512xf32>
      %301 = stablehlo.rsqrt %300 : tensor<512xf32>
      mhlo.return %301 : tensor<512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>) -> tensor<512xf32>
    %3 = "mhlo.fusion"() <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x32x32xf32>
      mhlo.return %299 : tensor<8x128x32x32xf32>
    }) {output_operand_aliasing = []} : () -> tensor<8x128x32x32xf32>
    %4 = "mhlo.fusion"(%arg215) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<256xf32>
      %301 = stablehlo.rsqrt %300 : tensor<256xf32>
      mhlo.return %301 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %5 = "mhlo.fusion"(%arg207) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<128xf32>
      mhlo.return %301 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %6 = "mhlo.fusion"(%arg205) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<128xf32>
      mhlo.return %301 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %7 = "mhlo.fusion"(%arg203) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<128xf32>
      mhlo.return %301 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %8 = "mhlo.fusion"(%arg201) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<64xf32>
      %301 = stablehlo.rsqrt %300 : tensor<64xf32>
      mhlo.return %301 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %9 = "mhlo.fusion"(%arg199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<64xf32>
      %301 = stablehlo.rsqrt %300 : tensor<64xf32>
      mhlo.return %301 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %10 = "mhlo.fusion"(%arg197) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<64xf32>
      %301 = stablehlo.rsqrt %300 : tensor<64xf32>
      mhlo.return %301 : tensor<64xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>) -> tensor<64xf32>
    %11 = "mhlo.fusion"(%arg195) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<32xf32>
      %301 = stablehlo.rsqrt %300 : tensor<32xf32>
      mhlo.return %301 : tensor<32xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>) -> tensor<32xf32>
    %12 = stablehlo.custom_call @__cudnn$convForward(%arg244, %arg0) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,32,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<8x3x256x256xf32>, tensor<32x3x3x3xf32>) -> tuple<tensor<8x32x128x128xf32>, tensor<0xui8>>
    %13 = stablehlo.get_tuple_element %12[0] : (tuple<tensor<8x32x128x128xf32>, tensor<0xui8>>) -> tensor<8x32x128x128xf32>
    %14 = "mhlo.fusion"(%arg2, %arg1, %11, %13, %arg194) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<32xf32>, %arg246: tensor<32xf32>, %arg247: tensor<32xf32>, %arg248: tensor<8x32x128x128xf32>, %arg249: tensor<32xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x32x128x128xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x32x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x32x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<32xf32>) -> tensor<8x32x128x128xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x32x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x32x128x128xf32>
      %308 = stablehlo.negate %306 : tensor<8x32x128x128xf32>
      %309 = stablehlo.exponential %308 : tensor<8x32x128x128xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x32x128x128xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x32x128x128xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x32x128x128xf32>
      mhlo.return %312 : tensor<8x32x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<8x32x128x128xf32>, tensor<32xf32>) -> tensor<8x32x128x128xf32>
    %15 = stablehlo.custom_call @__cudnn$convForward(%14, %arg3) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<8x32x128x128xf32>, tensor<64x32x1x1xf32>) -> tuple<tensor<8x64x128x128xf32>, tensor<0xui8>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<8x64x128x128xf32>, tensor<0xui8>>) -> tensor<8x64x128x128xf32>
    %17 = "mhlo.fusion"(%arg5, %arg4, %10, %16, %arg196) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<8x64x128x128xf32>, %arg249: tensor<64xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x64x128x128xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x64x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x64x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x64x128x128xf32>
      %308 = stablehlo.negate %306 : tensor<8x64x128x128xf32>
      %309 = stablehlo.exponential %308 : tensor<8x64x128x128xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x64x128x128xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x64x128x128xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x64x128x128xf32>
      mhlo.return %312 : tensor<8x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x64x128x128xf32>, tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %18 = stablehlo.custom_call @__cudnn$convForward(%17, %arg6) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x128x128xf32>, tensor<64x1x3x3xf32>) -> tuple<tensor<8x64x128x128xf32>, tensor<0xui8>>
    %19 = stablehlo.get_tuple_element %18[0] : (tuple<tensor<8x64x128x128xf32>, tensor<0xui8>>) -> tensor<8x64x128x128xf32>
    %20 = "mhlo.fusion"(%arg8, %arg7, %9, %19, %arg198) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<8x64x128x128xf32>, %arg249: tensor<64xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x64x128x128xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x64x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x64x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x64x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x64x128x128xf32>
      %308 = stablehlo.negate %306 : tensor<8x64x128x128xf32>
      %309 = stablehlo.exponential %308 : tensor<8x64x128x128xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x64x128x128xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x64x128x128xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x64x128x128xf32>
      mhlo.return %312 : tensor<8x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x64x128x128xf32>, tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %21 = stablehlo.custom_call @__cudnn$convForward(%20, %arg9) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,64,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x128x128xf32>, tensor<64x64x1x1xf32>) -> tuple<tensor<8x64x128x128xf32>, tensor<0xui8>>
    %22 = stablehlo.get_tuple_element %21[0] : (tuple<tensor<8x64x128x128xf32>, tensor<0xui8>>) -> tensor<8x64x128x128xf32>
    %23 = "mhlo.fusion"(%arg11, %arg10, %8, %22, %arg200) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<64xf32>, %arg246: tensor<64xf32>, %arg247: tensor<64xf32>, %arg248: tensor<8x64x128x128xf32>, %arg249: tensor<64xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x64x128x128xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x64x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x64x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<64xf32>) -> tensor<8x64x128x128xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x64x128x128xf32>
      mhlo.return %306 : tensor<8x64x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<8x64x128x128xf32>, tensor<64xf32>) -> tensor<8x64x128x128xf32>
    %24 = stablehlo.custom_call @__cudnn$convForward(%23, %arg12) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,128,128]{3,2,1,0}, u8[0]{0})"} : (tensor<8x64x128x128xf32>, tensor<128x64x1x1xf32>) -> tuple<tensor<8x128x128x128xf32>, tensor<0xui8>>
    %25 = stablehlo.get_tuple_element %24[0] : (tuple<tensor<8x128x128x128xf32>, tensor<0xui8>>) -> tensor<8x128x128x128xf32>
    %26 = "mhlo.fusion"(%arg14, %arg13, %7, %25, %arg202) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<8x128x128x128xf32>, %arg249: tensor<128xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x128x128x128xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x128x128x128xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x128x128x128xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x128x128xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x128x128x128xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x128x128xf32>
      %308 = stablehlo.negate %306 : tensor<8x128x128x128xf32>
      %309 = stablehlo.exponential %308 : tensor<8x128x128x128xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x128x128x128xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x128x128x128xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x128x128x128xf32>
      mhlo.return %312 : tensor<8x128x128x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<8x128x128x128xf32>, tensor<128xf32>) -> tensor<8x128x128x128xf32>
    %27 = stablehlo.custom_call @__cudnn$convForward(%26, %arg15) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x128x128xf32>, tensor<128x1x3x3xf32>) -> tuple<tensor<8x128x64x64xf32>, tensor<0xui8>>
    %28 = stablehlo.get_tuple_element %27[0] : (tuple<tensor<8x128x64x64xf32>, tensor<0xui8>>) -> tensor<8x128x64x64xf32>
    %29 = "mhlo.fusion"(%arg17, %arg16, %6, %28, %arg204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<8x128x64x64xf32>, %arg249: tensor<128xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x128x64x64xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x128x64x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x128x64x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x128x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x64x64xf32>
      %308 = stablehlo.negate %306 : tensor<8x128x64x64xf32>
      %309 = stablehlo.exponential %308 : tensor<8x128x64x64xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x128x64x64xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x128x64x64xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x128x64x64xf32>
      mhlo.return %312 : tensor<8x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<8x128x64x64xf32>, tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %30 = stablehlo.custom_call @__cudnn$convForward(%29, %arg18) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x64x64xf32>, tensor<128x128x1x1xf32>) -> tuple<tensor<8x128x64x64xf32>, tensor<0xui8>>
    %31 = stablehlo.get_tuple_element %30[0] : (tuple<tensor<8x128x64x64xf32>, tensor<0xui8>>) -> tensor<8x128x64x64xf32>
    %32 = "mhlo.fusion"(%arg20, %arg19, %5, %31, %arg206) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<8x128x64x64xf32>, %arg249: tensor<128xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x128x64x64xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x128x64x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x128x64x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x128x64x64xf32>
      mhlo.return %306 : tensor<8x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<8x128x64x64xf32>, tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %33 = "mhlo.fusion"(%arg213) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<128xf32>
      %301 = stablehlo.rsqrt %300 : tensor<128xf32>
      mhlo.return %301 : tensor<128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>) -> tensor<128xf32>
    %34 = "mhlo.fusion"(%arg211) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<256xf32>
      %301 = stablehlo.rsqrt %300 : tensor<256xf32>
      mhlo.return %301 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %35 = "mhlo.fusion"(%arg209) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>):
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %300 = stablehlo.add %arg245, %299 : tensor<256xf32>
      %301 = stablehlo.rsqrt %300 : tensor<256xf32>
      mhlo.return %301 : tensor<256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>) -> tensor<256xf32>
    %36 = stablehlo.custom_call @__cudnn$convForward(%32, %arg21) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<8x256x64x64xf32>, tensor<0xui8>>
    %37 = stablehlo.get_tuple_element %36[0] : (tuple<tensor<8x256x64x64xf32>, tensor<0xui8>>) -> tensor<8x256x64x64xf32>
    %38 = "mhlo.fusion"(%arg23, %arg22, %35, %37, %arg208) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8x256x64x64xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x256x64x64xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x256x64x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x256x64x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x64x64xf32>
      %308 = stablehlo.negate %306 : tensor<8x256x64x64xf32>
      %309 = stablehlo.exponential %308 : tensor<8x256x64x64xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x256x64x64xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x256x64x64xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x256x64x64xf32>
      mhlo.return %312 : tensor<8x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x64x64xf32>, tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %39 = stablehlo.custom_call @__cudnn$convForward(%38, %arg24) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<8x256x64x64xf32>, tensor<0xui8>>
    %40 = stablehlo.get_tuple_element %39[0] : (tuple<tensor<8x256x64x64xf32>, tensor<0xui8>>) -> tensor<8x256x64x64xf32>
    %41 = "mhlo.fusion"(%arg26, %arg25, %34, %40, %arg210) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8x256x64x64xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x256x64x64xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x256x64x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x256x64x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x64x64xf32>
      %308 = stablehlo.negate %306 : tensor<8x256x64x64xf32>
      %309 = stablehlo.exponential %308 : tensor<8x256x64x64xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x256x64x64xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x256x64x64xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x256x64x64xf32>
      mhlo.return %312 : tensor<8x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x64x64xf32>, tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %42 = stablehlo.custom_call @__cudnn$convForward(%41, %arg27) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x64x64xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<8x128x64x64xf32>, tensor<0xui8>>
    %43 = stablehlo.get_tuple_element %42[0] : (tuple<tensor<8x128x64x64xf32>, tensor<0xui8>>) -> tensor<8x128x64x64xf32>
    %44 = "mhlo.fusion"(%32, %arg29, %arg28, %33, %43, %arg212) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x128x64x64xf32>, %arg246: tensor<128xf32>, %arg247: tensor<128xf32>, %arg248: tensor<128xf32>, %arg249: tensor<8x128x64x64xf32>, %arg250: tensor<128xf32>):
      %299 = stablehlo.broadcast_in_dim %arg250, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %300 = stablehlo.subtract %arg249, %299 : tensor<8x128x64x64xf32>
      %301 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x128x64x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x128x64x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x64x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x128x64x64xf32>
      %307 = stablehlo.add %arg245, %306 : tensor<8x128x64x64xf32>
      mhlo.return %307 : tensor<8x128x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x64x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<8x128x64x64xf32>, tensor<128xf32>) -> tensor<8x128x64x64xf32>
    %45 = stablehlo.custom_call @__cudnn$convForward(%44, %arg30) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,64,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x64x64xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<8x256x64x64xf32>, tensor<0xui8>>
    %46 = stablehlo.get_tuple_element %45[0] : (tuple<tensor<8x256x64x64xf32>, tensor<0xui8>>) -> tensor<8x256x64x64xf32>
    %47 = "mhlo.fusion"(%arg32, %arg31, %4, %46, %arg214) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8x256x64x64xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x256x64x64xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x256x64x64xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x256x64x64xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x64x64xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x256x64x64xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x64x64xf32>
      %308 = stablehlo.negate %306 : tensor<8x256x64x64xf32>
      %309 = stablehlo.exponential %308 : tensor<8x256x64x64xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x256x64x64xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x256x64x64xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x256x64x64xf32>
      mhlo.return %312 : tensor<8x256x64x64xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8x256x64x64xf32>, tensor<256xf32>) -> tensor<8x256x64x64xf32>
    %48 = stablehlo.custom_call @__cudnn$convForward(%47, %arg33) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x64x64xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>
    %49 = stablehlo.get_tuple_element %48[0] : (tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>) -> tensor<8x256x32x32xf32>
    %50 = "mhlo.fusion"(%arg35, %arg34, %49, %arg216, %arg217) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<8x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<256xf32>
      %303 = stablehlo.rsqrt %302 : tensor<256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x256x32x32xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x256x32x32xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x256x32x32xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x32x32xf32>
      %311 = stablehlo.negate %309 : tensor<8x256x32x32xf32>
      %312 = stablehlo.exponential %311 : tensor<8x256x32x32xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x256x32x32xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x256x32x32xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x256x32x32xf32>
      mhlo.return %315 : tensor<8x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %51 = stablehlo.custom_call @__cudnn$convForward(%50, %arg36) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x32x32xf32>, tensor<256x256x1x1xf32>) -> tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>
    %52 = stablehlo.get_tuple_element %51[0] : (tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>) -> tensor<8x256x32x32xf32>
    %53 = "mhlo.fusion"(%arg38, %arg37, %52, %arg218, %arg219) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<8x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<256xf32>
      %303 = stablehlo.rsqrt %302 : tensor<256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x256x32x32xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x256x32x32xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x256x32x32xf32>
      mhlo.return %309 : tensor<8x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %54 = stablehlo.custom_call @__cudnn$convForward(%53, %arg39) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x32x32xf32>, tensor<256x1x3x3xf32>) -> tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>
    %55 = stablehlo.get_tuple_element %54[0] : (tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>) -> tensor<8x256x32x32xf32>
    %56 = "mhlo.fusion"(%arg41, %arg40, %55, %arg220, %arg221) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<8x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<256xf32>
      %303 = stablehlo.rsqrt %302 : tensor<256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x256x32x32xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x256x32x32xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x256x32x32xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x256x32x32xf32>
      %311 = stablehlo.negate %309 : tensor<8x256x32x32xf32>
      %312 = stablehlo.exponential %311 : tensor<8x256x32x32xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x256x32x32xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x256x32x32xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x256x32x32xf32>
      mhlo.return %315 : tensor<8x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %57 = stablehlo.custom_call @__cudnn$convForward(%56, %arg42) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x32x32xf32>, tensor<128x256x1x1xf32>) -> tuple<tensor<8x128x32x32xf32>, tensor<0xui8>>
    %58 = stablehlo.get_tuple_element %57[0] : (tuple<tensor<8x128x32x32xf32>, tensor<0xui8>>) -> tensor<8x128x32x32xf32>
    %59 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x32x32xf32>):
      %299 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %300 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %301 = stablehlo.add %299, %300 : tensor<2x16xi32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %303 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %304 = stablehlo.compare LT, %302, %303 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %305 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %306 = stablehlo.add %302, %305 : tensor<1x1x2x16xi32>
      %307 = stablehlo.select %304, %306, %302 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %310 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %311 = stablehlo.compare LT, %301, %310 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %312 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %313 = stablehlo.add %301, %312 : tensor<2x16xi32>
      %314 = stablehlo.select %311, %313, %301 : tensor<2x16xi1>, tensor<2x16xi32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %316 = stablehlo.concatenate %309, %315, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %317 = mhlo.bitcast %316 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %318 = "stablehlo.gather"(%arg245, %317) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 128, 1, 1>}> : (tensor<8x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x8x128x1x1xf32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x8x128x1x1xf32>) -> tensor<2x16x2x16x8x128xf32>
      %320 = stablehlo.transpose %319, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x16x2x16x8x128xf32>) -> tensor<8x128x2x2x16x16xf32>
      mhlo.return %320 : tensor<8x128x2x2x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x32x32xf32>) -> tensor<8x128x2x2x16x16xf32>
    %60 = "mhlo.fusion"(%58) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x32x32xf32>):
      %299 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %300 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %301 = stablehlo.add %299, %300 : tensor<2x16xi32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %303 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %304 = stablehlo.compare LT, %302, %303 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %305 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %306 = stablehlo.add %302, %305 : tensor<1x1x2x16xi32>
      %307 = stablehlo.select %304, %306, %302 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %310 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %311 = stablehlo.compare LT, %301, %310 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %312 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %313 = stablehlo.add %301, %312 : tensor<2x16xi32>
      %314 = stablehlo.select %311, %313, %301 : tensor<2x16xi1>, tensor<2x16xi32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %316 = stablehlo.concatenate %309, %315, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %317 = mhlo.bitcast %316 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %318 = "stablehlo.gather"(%arg245, %317) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 128, 1, 1>}> : (tensor<8x128x32x32xf32>, tensor<1024x2xi32>) -> tensor<1024x8x128x1x1xf32>
      %319 = mhlo.bitcast %318 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<1024x8x128x1x1xf32>) -> tensor<1024x8x128xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %320 = stablehlo.reduce(%319 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1024x8x128xf32>, tensor<f32>) -> tensor<1024x8xf32>
      mhlo.return %320 : tensor<1024x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x32x32xf32>) -> tensor<1024x8xf32>
    %61 = "mhlo.fusion"(%60) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<1024x8xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x8xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<1024x8xf32>) -> tensor<8xf32>
    %62 = "mhlo.fusion"(%59, %61) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x2x2x16x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x128x2x2x16x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x128x2x2x16x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x128x2x2x16x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x128x2x2x16x16xf32>) -> tensor<8x256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %305 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x2x2x16x16xf32>, tensor<8xf32>) -> tensor<8x256xf32>
    %63 = "mhlo.fusion"(%62) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %64 = "mhlo.fusion"(%arg43, %arg44, %59, %63, %61) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<8x128x2x2x16x16xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<1x8x131072xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %302 = stablehlo.multiply %arg249, %301 : tensor<8xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0] : (tensor<8xf32>) -> tensor<8x128x2x2x16x16xf32>
      %304 = stablehlo.subtract %arg247, %303 : tensor<8x128x2x2x16x16xf32>
      %305 = stablehlo.multiply %arg248, %301 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %307 = stablehlo.add %305, %306 : tensor<8xf32>
      %308 = stablehlo.rsqrt %307 : tensor<8xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0] : (tensor<8xf32>) -> tensor<8x128x2x2x16x16xf32>
      %310 = stablehlo.multiply %304, %309 : tensor<8x128x2x2x16x16xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x128x2x2x16x16xf32>) -> tensor<1x8x131072xf32>
      %312 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<1x8x131072xf32>
      %314 = stablehlo.multiply %311, %313 : tensor<1x8x131072xf32>
      %315 = stablehlo.add %300, %314 : tensor<1x8x131072xf32>
      mhlo.return %315 : tensor<1x8x131072xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<8x128x2x2x16x16xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x131072xf32>
    %65 = mhlo.bitcast %64 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x131072xf32>) -> tensor<8x128x4x256xf32>
    %66 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%65, %arg45, %arg46) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<8x257x4x256xf32>, tensor<0xui8>>
    %67 = stablehlo.get_tuple_element %66[0] : (tuple<tensor<8x257x4x256xf32>, tensor<0xui8>>) -> tensor<8x257x4x256xf32>
    %68 = "mhlo.fusion"(%67) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x257x4x256xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 129:257, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x4x256xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x128x4x256xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:129, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x1x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x256xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x256xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x256xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x256xf32>) -> tensor<8x128x4x256xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x128x4x256xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x128x4x256xf32>, tensor<f32>) -> tensor<8x128x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x128x4xf32>) -> tensor<8x128x4x256xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x128x4x256xf32>
      mhlo.return %318 : tensor<8x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
    %69 = mhlo.bitcast %59 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x128x2x2x16x16xf32>) -> tensor<8x128x4x256xf32>
    %70 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%68, %arg47, %arg48, %69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>) -> tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>
    %71 = stablehlo.get_tuple_element %70[0] : (tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>) -> tensor<8x128x4x256xf32>
    %72 = "mhlo.fusion"(%71) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %300 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>) -> tensor<8x256xf32>
    %73 = "mhlo.fusion"(%72) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %74 = "mhlo.fusion"(%71, %73) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x128x4x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x128x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %305 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>, tensor<8xf32>) -> tensor<8x256xf32>
    %75 = "mhlo.fusion"(%74) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %76 = "mhlo.fusion"(%arg49, %arg50, %71, %75, %73) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<8x128x4x256xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x128x4x256xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x128x4x256xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x128x4x256xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x128x4x256xf32>
      mhlo.return %312 : tensor<8x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %77 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%76, %arg51, %arg52) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<8x256x4x256xf32>, tensor<0xui8>>
    %78 = stablehlo.get_tuple_element %77[0] : (tuple<tensor<8x256x4x256xf32>, tensor<0xui8>>) -> tensor<8x256x4x256xf32>
    %79 = "mhlo.fusion"(%78) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x256xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x256x4x256xf32>
      %301 = stablehlo.exponential %300 : tensor<8x256x4x256xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x256x4x256xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x256x4x256xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x256x4x256xf32>
      mhlo.return %304 : tensor<8x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x256xf32>) -> tensor<8x256x4x256xf32>
    %80 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%79, %arg53, %arg54, %71) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>) -> tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>
    %81 = stablehlo.get_tuple_element %80[0] : (tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>) -> tensor<8x128x4x256xf32>
    %82 = "mhlo.fusion"(%81) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %300 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>) -> tensor<8x256xf32>
    %83 = "mhlo.fusion"(%82) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %84 = "mhlo.fusion"(%81, %83) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x128x4x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x128x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %305 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>, tensor<8xf32>) -> tensor<8x256xf32>
    %85 = "mhlo.fusion"(%84) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %86 = "mhlo.fusion"(%arg55, %arg56, %81, %85, %83) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<8x128x4x256xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x128x4x256xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x128x4x256xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x128x4x256xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x128x4x256xf32>
      mhlo.return %312 : tensor<8x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %87 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%86, %arg57, %arg58) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,257,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x4x256xf32>, tensor<257x128x1x1xf32>, tensor<257xf32>) -> tuple<tensor<8x257x4x256xf32>, tensor<0xui8>>
    %88 = stablehlo.get_tuple_element %87[0] : (tuple<tensor<8x257x4x256xf32>, tensor<0xui8>>) -> tensor<8x257x4x256xf32>
    %89 = "mhlo.fusion"(%88) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x257x4x256xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 129:257, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x128x4x256xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x128x4x256xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:129, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:256] : (tensor<8x257x4x256xf32>) -> tensor<8x1x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x256xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x256xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x256xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x256xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x256xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x256xf32>) -> tensor<8x4x256xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x256xf32>) -> tensor<8x128x4x256xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x128x4x256xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x128x4x256xf32>, tensor<f32>) -> tensor<8x128x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x128x4xf32>) -> tensor<8x128x4x256xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x128x4x256xf32>
      mhlo.return %318 : tensor<8x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x257x4x256xf32>) -> tensor<8x128x4x256xf32>
    %90 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%89, %arg59, %arg60, %81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x4x256xf32>, tensor<128x128x1x1xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>) -> tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>
    %91 = stablehlo.get_tuple_element %90[0] : (tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>) -> tensor<8x128x4x256xf32>
    %92 = "mhlo.fusion"(%91) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %300 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>) -> tensor<8x256xf32>
    %93 = "mhlo.fusion"(%92) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %94 = "mhlo.fusion"(%91, %93) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x128x4x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x128x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %305 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>, tensor<8xf32>) -> tensor<8x256xf32>
    %95 = "mhlo.fusion"(%94) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %96 = "mhlo.fusion"(%arg61, %arg62, %91, %95, %93) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<8x128x4x256xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x128x4x256xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x128x4x256xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x128x4x256xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x128x4x256xf32>
      mhlo.return %312 : tensor<8x128x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x128x4x256xf32>
    %97 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%96, %arg63, %arg64) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x4x256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tuple<tensor<8x256x4x256xf32>, tensor<0xui8>>
    %98 = stablehlo.get_tuple_element %97[0] : (tuple<tensor<8x256x4x256xf32>, tensor<0xui8>>) -> tensor<8x256x4x256xf32>
    %99 = "mhlo.fusion"(%98) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x256xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x256xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x256x4x256xf32>
      %301 = stablehlo.exponential %300 : tensor<8x256x4x256xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x256x4x256xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x256x4x256xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x256x4x256xf32>
      mhlo.return %304 : tensor<8x256x4x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x256xf32>) -> tensor<8x256x4x256xf32>
    %100 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%99, %arg65, %arg66, %91) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,128,4,256]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>) -> tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>
    %101 = stablehlo.get_tuple_element %100[0] : (tuple<tensor<8x128x4x256xf32>, tensor<0xui8>>) -> tensor<8x128x4x256xf32>
    %102 = "mhlo.fusion"(%101) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %300 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>) -> tensor<8x256xf32>
    %103 = "mhlo.fusion"(%102) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %104 = "mhlo.fusion"(%101, %103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x4x256xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x128x4x256xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x128x4x256xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x256x512xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x256x512xf32>, tensor<f32>) -> tensor<8x256xf32>
      mhlo.return %305 : tensor<8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x4x256xf32>, tensor<8xf32>) -> tensor<8x256xf32>
    %105 = "mhlo.fusion"(%104) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x256xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256xf32>) -> tensor<8xf32>
    %106 = "mhlo.fusion"(%arg67, %arg68, %101, %105, %103) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<128xf32>, %arg246: tensor<128xf32>, %arg247: tensor<8x128x4x256xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %cst = stablehlo.constant dense<7.62939453E-6> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x128x4x256xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x128x4x256xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x128x4x256xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<128xf32>) -> tensor<8x128x4x256xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x128x4x256xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x128x4x256xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x128x4x256xf32>) -> tensor<8x128x2x2x16x16xf32>
      %314 = stablehlo.transpose %313, dims = [2, 4, 3, 5, 0, 1] : (tensor<8x128x2x2x16x16xf32>) -> tensor<2x16x2x16x8x128xf32>
      mhlo.return %314 : tensor<2x16x2x16x8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<128xf32>, tensor<128xf32>, tensor<8x128x4x256xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x16x2x16x8x128xf32>
    %107 = "mhlo.fusion"(%3, %106) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128x32x32xf32>, %arg246: tensor<2x16x2x16x8x128xf32>):
      %299 = stablehlo.iota dim = 1 : tensor<2x16xi32>
      %300 = stablehlo.iota dim = 0 : tensor<2x16xi32>
      %301 = stablehlo.add %299, %300 : tensor<2x16xi32>
      %302 = mhlo.bitcast %301 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x16xi32>) -> tensor<1x1x2x16xi32>
      %c = stablehlo.constant dense<0> : tensor<i32>
      %303 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %304 = stablehlo.compare LT, %302, %303 : (tensor<1x1x2x16xi32>, tensor<1x1x2x16xi32>) -> tensor<1x1x2x16xi1>
      %c_0 = stablehlo.constant dense<32> : tensor<i32>
      %305 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x16xi32>
      %306 = stablehlo.add %302, %305 : tensor<1x1x2x16xi32>
      %307 = stablehlo.select %304, %306, %302 : tensor<1x1x2x16xi1>, tensor<1x1x2x16xi32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x16xi32>) -> tensor<2x16xi32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0, 1] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %310 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %311 = stablehlo.compare LT, %301, %310 : (tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi1>
      %312 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x16xi32>
      %313 = stablehlo.add %301, %312 : tensor<2x16xi32>
      %314 = stablehlo.select %311, %313, %301 : tensor<2x16xi1>, tensor<2x16xi32>
      %315 = stablehlo.broadcast_in_dim %314, dims = [2, 3] : (tensor<2x16xi32>) -> tensor<2x16x2x16x1xi32>
      %316 = stablehlo.concatenate %309, %315, dim = 4 : (tensor<2x16x2x16x1xi32>, tensor<2x16x2x16x1xi32>) -> tensor<2x16x2x16x2xi32>
      %317 = mhlo.bitcast %316 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x16x2x16x2xi32>) -> tensor<1024x2xi32>
      %318 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x16x2x16x8x128xf32>) -> tensor<1024x8x128x1x1xf32>
      %319 = "stablehlo.scatter"(%arg245, %317, %318) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %320 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %320 : tensor<f32>
      }) : (tensor<8x128x32x32xf32>, tensor<1024x2xi32>, tensor<1024x8x128x1x1xf32>) -> tensor<8x128x32x32xf32>
      mhlo.return %319 : tensor<8x128x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128x32x32xf32>, tensor<2x16x2x16x8x128xf32>) -> tensor<8x128x32x32xf32>
    %108 = stablehlo.custom_call @__cudnn$convForward(%107, %arg69) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<8x128x32x32xf32>, tensor<256x128x1x1xf32>) -> tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>
    %109 = stablehlo.get_tuple_element %108[0] : (tuple<tensor<8x256x32x32xf32>, tensor<0xui8>>) -> tensor<8x256x32x32xf32>
    %110 = "mhlo.fusion"(%arg71, %arg70, %109, %arg222, %arg223) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<256xf32>, %arg247: tensor<8x256x32x32xf32>, %arg248: tensor<256xf32>, %arg249: tensor<256xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x256x32x32xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<256xf32>
      %303 = stablehlo.rsqrt %302 : tensor<256xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x256x32x32xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x256x32x32xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x32x32xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x256x32x32xf32>
      mhlo.return %309 : tensor<8x256x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<256xf32>, tensor<8x256x32x32xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<8x256x32x32xf32>
    %111 = stablehlo.custom_call @__cudnn$convForward(%110, %arg72) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,32,32]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x32x32xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<8x512x32x32xf32>, tensor<0xui8>>
    %112 = stablehlo.get_tuple_element %111[0] : (tuple<tensor<8x512x32x32xf32>, tensor<0xui8>>) -> tensor<8x512x32x32xf32>
    %113 = "mhlo.fusion"(%arg74, %arg73, %2, %112, %arg224) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<512xf32>, %arg248: tensor<8x512x32x32xf32>, %arg249: tensor<512xf32>):
      %299 = stablehlo.broadcast_in_dim %arg249, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
      %300 = stablehlo.subtract %arg248, %299 : tensor<8x512x32x32xf32>
      %301 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
      %302 = stablehlo.multiply %300, %301 : tensor<8x512x32x32xf32>
      %303 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
      %304 = stablehlo.multiply %302, %303 : tensor<8x512x32x32xf32>
      %305 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<8x512x32x32xf32>
      %306 = stablehlo.add %304, %305 : tensor<8x512x32x32xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %307 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x32x32xf32>
      %308 = stablehlo.negate %306 : tensor<8x512x32x32xf32>
      %309 = stablehlo.exponential %308 : tensor<8x512x32x32xf32>
      %310 = stablehlo.add %307, %309 : tensor<8x512x32x32xf32>
      %311 = stablehlo.divide %307, %310 : tensor<8x512x32x32xf32>
      %312 = stablehlo.multiply %306, %311 : tensor<8x512x32x32xf32>
      mhlo.return %312 : tensor<8x512x32x32xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<8x512x32x32xf32>, tensor<512xf32>) -> tensor<8x512x32x32xf32>
    %114 = stablehlo.custom_call @__cudnn$convForward(%113, %arg75) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x32x32xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<8x512x16x16xf32>, tensor<0xui8>>
    %115 = stablehlo.get_tuple_element %114[0] : (tuple<tensor<8x512x16x16xf32>, tensor<0xui8>>) -> tensor<8x512x16x16xf32>
    %116 = "mhlo.fusion"(%arg77, %arg76, %115, %arg226, %arg227) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<8x512x16x16xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x512x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<512xf32>
      %303 = stablehlo.rsqrt %302 : tensor<512xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x512x16x16xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x512x16x16xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<8x512x16x16xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x512x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x16x16xf32>
      %311 = stablehlo.negate %309 : tensor<8x512x16x16xf32>
      %312 = stablehlo.exponential %311 : tensor<8x512x16x16xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x512x16x16xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x512x16x16xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x512x16x16xf32>
      mhlo.return %315 : tensor<8x512x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x16x16xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x16x16xf32>
    %117 = stablehlo.custom_call @__cudnn$convForward(%116, %arg78) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x16x16xf32>, tensor<384x512x1x1xf32>) -> tuple<tensor<8x384x16x16xf32>, tensor<0xui8>>
    %118 = stablehlo.get_tuple_element %117[0] : (tuple<tensor<8x384x16x16xf32>, tensor<0xui8>>) -> tensor<8x384x16x16xf32>
    %119 = "mhlo.fusion"(%arg80, %arg79, %118, %arg228, %arg229) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<8x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<384xf32>
      %303 = stablehlo.rsqrt %302 : tensor<384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x384x16x16xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x384x16x16xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x384x16x16xf32>
      mhlo.return %309 : tensor<8x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %120 = stablehlo.custom_call @__cudnn$convForward(%119, %arg81) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x16x16xf32>, tensor<384x1x3x3xf32>) -> tuple<tensor<8x384x16x16xf32>, tensor<0xui8>>
    %121 = stablehlo.get_tuple_element %120[0] : (tuple<tensor<8x384x16x16xf32>, tensor<0xui8>>) -> tensor<8x384x16x16xf32>
    %122 = "mhlo.fusion"(%arg83, %arg82, %121, %arg230, %arg231) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<8x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<384xf32>
      %303 = stablehlo.rsqrt %302 : tensor<384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x384x16x16xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x384x16x16xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x384x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x384x16x16xf32>
      %311 = stablehlo.negate %309 : tensor<8x384x16x16xf32>
      %312 = stablehlo.exponential %311 : tensor<8x384x16x16xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x384x16x16xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x384x16x16xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x384x16x16xf32>
      mhlo.return %315 : tensor<8x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %123 = stablehlo.custom_call @__cudnn$convForward(%122, %arg84) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x16x16xf32>, tensor<192x384x1x1xf32>) -> tuple<tensor<8x192x16x16xf32>, tensor<0xui8>>
    %124 = stablehlo.get_tuple_element %123[0] : (tuple<tensor<8x192x16x16xf32>, tensor<0xui8>>) -> tensor<8x192x16x16xf32>
    %125 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x16x16xf32>):
      %299 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %300 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %301 = stablehlo.multiply %299, %300 : tensor<8xi32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %303 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %304 = stablehlo.add %302, %303 : tensor<2x8xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %307 = stablehlo.compare LT, %305, %306 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %309 = stablehlo.add %305, %308 : tensor<1x1x2x8xi32>
      %310 = stablehlo.select %307, %309, %305 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %313 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %314 = stablehlo.compare LT, %304, %313 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %315 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %316 = stablehlo.add %304, %315 : tensor<2x8xi32>
      %317 = stablehlo.select %314, %316, %304 : tensor<2x8xi1>, tensor<2x8xi32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %319 = stablehlo.concatenate %312, %318, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %321 = "stablehlo.gather"(%arg245, %320) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 192, 1, 1>}> : (tensor<8x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x8x192x1x1xf32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x8x192x1x1xf32>) -> tensor<2x8x2x8x8x192xf32>
      %323 = stablehlo.transpose %322, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x8x2x8x8x192xf32>) -> tensor<8x192x2x2x8x8xf32>
      mhlo.return %323 : tensor<8x192x2x2x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x16x16xf32>) -> tensor<8x192x2x2x8x8xf32>
    %126 = "mhlo.fusion"(%124) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x16x16xf32>):
      %299 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %300 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %301 = stablehlo.multiply %299, %300 : tensor<8xi32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %303 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %304 = stablehlo.add %302, %303 : tensor<2x8xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %307 = stablehlo.compare LT, %305, %306 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %309 = stablehlo.add %305, %308 : tensor<1x1x2x8xi32>
      %310 = stablehlo.select %307, %309, %305 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %313 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %314 = stablehlo.compare LT, %304, %313 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %315 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %316 = stablehlo.add %304, %315 : tensor<2x8xi32>
      %317 = stablehlo.select %314, %316, %304 : tensor<2x8xi1>, tensor<2x8xi32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %319 = stablehlo.concatenate %312, %318, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %321 = "stablehlo.gather"(%arg245, %320) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 192, 1, 1>}> : (tensor<8x192x16x16xf32>, tensor<256x2xi32>) -> tensor<256x8x192x1x1xf32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<256x8x192x1x1xf32>) -> tensor<256x8x192xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %323 = stablehlo.reduce(%322 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<256x8x192xf32>, tensor<f32>) -> tensor<256x8xf32>
      mhlo.return %323 : tensor<256x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x16x16xf32>) -> tensor<256x8xf32>
    %127 = "mhlo.fusion"(%126) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<256x8xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<256x8xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<256x8xf32>) -> tensor<8xf32>
    %128 = "mhlo.fusion"(%125, %127) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x2x2x8x8xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x2x2x8x8xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x2x2x8x8xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x2x2x8x8xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x192x2x2x8x8xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x2x2x8x8xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %129 = "mhlo.fusion"(%128) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %130 = "mhlo.fusion"(%arg85, %arg86, %125, %129, %127) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x2x2x8x8xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<1x8x49152xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %302 = stablehlo.multiply %arg249, %301 : tensor<8xf32>
      %303 = stablehlo.broadcast_in_dim %302, dims = [0] : (tensor<8xf32>) -> tensor<8x192x2x2x8x8xf32>
      %304 = stablehlo.subtract %arg247, %303 : tensor<8x192x2x2x8x8xf32>
      %305 = stablehlo.multiply %arg248, %301 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %306 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %307 = stablehlo.add %305, %306 : tensor<8xf32>
      %308 = stablehlo.rsqrt %307 : tensor<8xf32>
      %309 = stablehlo.broadcast_in_dim %308, dims = [0] : (tensor<8xf32>) -> tensor<8x192x2x2x8x8xf32>
      %310 = stablehlo.multiply %304, %309 : tensor<8x192x2x2x8x8xf32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x192x2x2x8x8xf32>) -> tensor<1x8x49152xf32>
      %312 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<1x8x49152xf32>
      %314 = stablehlo.multiply %311, %313 : tensor<1x8x49152xf32>
      %315 = stablehlo.add %300, %314 : tensor<1x8x49152xf32>
      mhlo.return %315 : tensor<1x8x49152xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x2x2x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x49152xf32>
    %131 = mhlo.bitcast %130 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x49152xf32>) -> tensor<8x192x4x64xf32>
    %132 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%131, %arg87, %arg88) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>
    %133 = stablehlo.get_tuple_element %132[0] : (tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>) -> tensor<8x385x4x64xf32>
    %134 = "mhlo.fusion"(%133) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x385x4x64xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x192x4x64xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x64xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x64xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x192x4x64xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x192x4x64xf32>
      mhlo.return %318 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %135 = mhlo.bitcast %125 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x192x2x2x8x8xf32>) -> tensor<8x192x4x64xf32>
    %136 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%134, %arg89, %arg90, %135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %137 = stablehlo.get_tuple_element %136[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %138 = "mhlo.fusion"(%137) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %139 = "mhlo.fusion"(%138) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %140 = "mhlo.fusion"(%137, %139) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %141 = "mhlo.fusion"(%140) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %142 = "mhlo.fusion"(%arg91, %arg92, %137, %141, %139) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %143 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%142, %arg93, %arg94) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>
    %144 = stablehlo.get_tuple_element %143[0] : (tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>) -> tensor<8x384x4x64xf32>
    %145 = "mhlo.fusion"(%144) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x384x4x64xf32>
      %301 = stablehlo.exponential %300 : tensor<8x384x4x64xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x384x4x64xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x384x4x64xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x384x4x64xf32>
      mhlo.return %304 : tensor<8x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32>
    %146 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%145, %arg95, %arg96, %137) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %147 = stablehlo.get_tuple_element %146[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %148 = "mhlo.fusion"(%147) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %149 = "mhlo.fusion"(%148) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %150 = "mhlo.fusion"(%147, %149) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %151 = "mhlo.fusion"(%150) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %152 = "mhlo.fusion"(%arg97, %arg98, %147, %151, %149) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %153 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%152, %arg99, %arg100) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>
    %154 = stablehlo.get_tuple_element %153[0] : (tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>) -> tensor<8x385x4x64xf32>
    %155 = "mhlo.fusion"(%154) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x385x4x64xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x192x4x64xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x64xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x64xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x192x4x64xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x192x4x64xf32>
      mhlo.return %318 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %156 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%155, %arg101, %arg102, %147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %157 = stablehlo.get_tuple_element %156[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %158 = "mhlo.fusion"(%157) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %159 = "mhlo.fusion"(%158) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %160 = "mhlo.fusion"(%157, %159) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %161 = "mhlo.fusion"(%160) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %162 = "mhlo.fusion"(%arg103, %arg104, %157, %161, %159) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %163 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%162, %arg105, %arg106) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>
    %164 = stablehlo.get_tuple_element %163[0] : (tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>) -> tensor<8x384x4x64xf32>
    %165 = "mhlo.fusion"(%164) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x384x4x64xf32>
      %301 = stablehlo.exponential %300 : tensor<8x384x4x64xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x384x4x64xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x384x4x64xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x384x4x64xf32>
      mhlo.return %304 : tensor<8x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32>
    %166 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%165, %arg107, %arg108, %157) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %167 = stablehlo.get_tuple_element %166[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %168 = "mhlo.fusion"(%167) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %169 = "mhlo.fusion"(%168) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %170 = "mhlo.fusion"(%167, %169) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %171 = "mhlo.fusion"(%170) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %172 = "mhlo.fusion"(%arg109, %arg110, %167, %171, %169) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %173 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%172, %arg111, %arg112) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>
    %174 = stablehlo.get_tuple_element %173[0] : (tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>) -> tensor<8x385x4x64xf32>
    %175 = "mhlo.fusion"(%174) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x385x4x64xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x192x4x64xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x64xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x64xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x192x4x64xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x192x4x64xf32>
      mhlo.return %318 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %176 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%175, %arg113, %arg114, %167) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %177 = stablehlo.get_tuple_element %176[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %178 = "mhlo.fusion"(%177) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %179 = "mhlo.fusion"(%178) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %180 = "mhlo.fusion"(%177, %179) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %181 = "mhlo.fusion"(%180) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %182 = "mhlo.fusion"(%arg115, %arg116, %177, %181, %179) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %183 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%182, %arg117, %arg118) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>
    %184 = stablehlo.get_tuple_element %183[0] : (tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>) -> tensor<8x384x4x64xf32>
    %185 = "mhlo.fusion"(%184) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x384x4x64xf32>
      %301 = stablehlo.exponential %300 : tensor<8x384x4x64xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x384x4x64xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x384x4x64xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x384x4x64xf32>
      mhlo.return %304 : tensor<8x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32>
    %186 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%185, %arg119, %arg120, %177) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %187 = stablehlo.get_tuple_element %186[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %188 = "mhlo.fusion"(%187) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %189 = "mhlo.fusion"(%188) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %190 = "mhlo.fusion"(%187, %189) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %191 = "mhlo.fusion"(%190) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %192 = "mhlo.fusion"(%arg121, %arg122, %187, %191, %189) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %193 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%192, %arg123, %arg124) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,385,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<385x192x1x1xf32>, tensor<385xf32>) -> tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>
    %194 = stablehlo.get_tuple_element %193[0] : (tuple<tensor<8x385x4x64xf32>, tensor<0xui8>>) -> tensor<8x385x4x64xf32>
    %195 = "mhlo.fusion"(%194) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x385x4x64xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 193:385, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x192x4x64xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x192x4x64xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:193, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:64] : (tensor<8x385x4x64xf32>) -> tensor<8x1x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x64xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x64xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x64xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x64xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x64xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x64xf32>) -> tensor<8x4x64xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x64xf32>) -> tensor<8x192x4x64xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x192x4x64xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x192x4x64xf32>, tensor<f32>) -> tensor<8x192x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x192x4xf32>) -> tensor<8x192x4x64xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x192x4x64xf32>
      mhlo.return %318 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x385x4x64xf32>) -> tensor<8x192x4x64xf32>
    %196 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%195, %arg125, %arg126, %187) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<192x192x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %197 = stablehlo.get_tuple_element %196[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %198 = "mhlo.fusion"(%197) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %199 = "mhlo.fusion"(%198) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %200 = "mhlo.fusion"(%197, %199) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %201 = "mhlo.fusion"(%200) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %202 = "mhlo.fusion"(%arg127, %arg128, %197, %201, %199) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      mhlo.return %312 : tensor<8x192x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x192x4x64xf32>
    %203 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%202, %arg129, %arg130) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x4x64xf32>, tensor<384x192x1x1xf32>, tensor<384xf32>) -> tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>
    %204 = stablehlo.get_tuple_element %203[0] : (tuple<tensor<8x384x4x64xf32>, tensor<0xui8>>) -> tensor<8x384x4x64xf32>
    %205 = "mhlo.fusion"(%204) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x384x4x64xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x384x4x64xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x384x4x64xf32>
      %301 = stablehlo.exponential %300 : tensor<8x384x4x64xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x384x4x64xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x384x4x64xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x384x4x64xf32>
      mhlo.return %304 : tensor<8x384x4x64xf32>
    }) {output_operand_aliasing = []} : (tensor<8x384x4x64xf32>) -> tensor<8x384x4x64xf32>
    %206 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%205, %arg131, %arg132, %197) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,192,4,64]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x4x64xf32>, tensor<192x384x1x1xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>) -> tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>
    %207 = stablehlo.get_tuple_element %206[0] : (tuple<tensor<8x192x4x64xf32>, tensor<0xui8>>) -> tensor<8x192x4x64xf32>
    %208 = "mhlo.fusion"(%207) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %300 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>) -> tensor<8x128xf32>
    %209 = "mhlo.fusion"(%208) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %210 = "mhlo.fusion"(%207, %209) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x4x64xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x192x4x64xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x192x4x64xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x128x384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x128x384xf32>, tensor<f32>) -> tensor<8x128xf32>
      mhlo.return %305 : tensor<8x128xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x4x64xf32>, tensor<8xf32>) -> tensor<8x128xf32>
    %211 = "mhlo.fusion"(%210) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x128xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x128xf32>) -> tensor<8xf32>
    %212 = "mhlo.fusion"(%arg133, %arg134, %207, %211, %209) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<192xf32>, %arg246: tensor<192xf32>, %arg247: tensor<8x192x4x64xf32>, %arg248: tensor<8xf32>, %arg249: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %cst = stablehlo.constant dense<2.03450527E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg249, %300 : tensor<8xf32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %303 = stablehlo.subtract %arg247, %302 : tensor<8x192x4x64xf32>
      %304 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %305 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %306 = stablehlo.add %304, %305 : tensor<8xf32>
      %307 = stablehlo.rsqrt %306 : tensor<8xf32>
      %308 = stablehlo.broadcast_in_dim %307, dims = [0] : (tensor<8xf32>) -> tensor<8x192x4x64xf32>
      %309 = stablehlo.multiply %303, %308 : tensor<8x192x4x64xf32>
      %310 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<192xf32>) -> tensor<8x192x4x64xf32>
      %311 = stablehlo.multiply %309, %310 : tensor<8x192x4x64xf32>
      %312 = stablehlo.add %299, %311 : tensor<8x192x4x64xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x192x4x64xf32>) -> tensor<8x192x2x2x8x8xf32>
      %314 = stablehlo.transpose %313, dims = [2, 4, 3, 5, 0, 1] : (tensor<8x192x2x2x8x8xf32>) -> tensor<2x8x2x8x8x192xf32>
      mhlo.return %314 : tensor<2x8x2x8x8x192xf32>
    }) {output_operand_aliasing = []} : (tensor<192xf32>, tensor<192xf32>, tensor<8x192x4x64xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x8x2x8x8x192xf32>
    %213 = "mhlo.fusion"(%1, %212) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x192x16x16xf32>, %arg246: tensor<2x8x2x8x8x192xf32>):
      %299 = stablehlo.iota dim = 0 : tensor<8xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %300 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8xi32>
      %301 = stablehlo.multiply %299, %300 : tensor<8xi32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<8xi32>) -> tensor<2x8xi32>
      %303 = stablehlo.iota dim = 0 : tensor<2x8xi32>
      %304 = stablehlo.add %302, %303 : tensor<2x8xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x8xi32>) -> tensor<1x1x2x8xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %307 = stablehlo.compare LT, %305, %306 : (tensor<1x1x2x8xi32>, tensor<1x1x2x8xi32>) -> tensor<1x1x2x8xi1>
      %c_1 = stablehlo.constant dense<16> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x8xi32>
      %309 = stablehlo.add %305, %308 : tensor<1x1x2x8xi32>
      %310 = stablehlo.select %307, %309, %305 : tensor<1x1x2x8xi1>, tensor<1x1x2x8xi32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x8xi32>) -> tensor<2x8xi32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %313 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %314 = stablehlo.compare LT, %304, %313 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
      %315 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x8xi32>
      %316 = stablehlo.add %304, %315 : tensor<2x8xi32>
      %317 = stablehlo.select %314, %316, %304 : tensor<2x8xi1>, tensor<2x8xi32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [2, 3] : (tensor<2x8xi32>) -> tensor<2x8x2x8x1xi32>
      %319 = stablehlo.concatenate %312, %318, dim = 4 : (tensor<2x8x2x8x1xi32>, tensor<2x8x2x8x1xi32>) -> tensor<2x8x2x8x2xi32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x8x2x8x2xi32>) -> tensor<256x2xi32>
      %321 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x8x2x8x8x192xf32>) -> tensor<256x8x192x1x1xf32>
      %322 = "stablehlo.scatter"(%arg245, %320, %321) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %323 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %323 : tensor<f32>
      }) : (tensor<8x192x16x16xf32>, tensor<256x2xi32>, tensor<256x8x192x1x1xf32>) -> tensor<8x192x16x16xf32>
      mhlo.return %322 : tensor<8x192x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x192x16x16xf32>, tensor<2x8x2x8x8x192xf32>) -> tensor<8x192x16x16xf32>
    %214 = stablehlo.custom_call @__cudnn$convForward(%213, %arg135) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,384,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x192x16x16xf32>, tensor<384x192x1x1xf32>) -> tuple<tensor<8x384x16x16xf32>, tensor<0xui8>>
    %215 = stablehlo.get_tuple_element %214[0] : (tuple<tensor<8x384x16x16xf32>, tensor<0xui8>>) -> tensor<8x384x16x16xf32>
    %216 = "mhlo.fusion"(%arg137, %arg136, %215, %arg232, %arg233) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<384xf32>, %arg246: tensor<384xf32>, %arg247: tensor<8x384x16x16xf32>, %arg248: tensor<384xf32>, %arg249: tensor<384xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x384x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<384xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<384xf32>
      %303 = stablehlo.rsqrt %302 : tensor<384xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x384x16x16xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x384x16x16xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<384xf32>) -> tensor<8x384x16x16xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x384x16x16xf32>
      mhlo.return %309 : tensor<8x384x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<384xf32>, tensor<384xf32>, tensor<8x384x16x16xf32>, tensor<384xf32>, tensor<384xf32>) -> tensor<8x384x16x16xf32>
    %217 = stablehlo.custom_call @__cudnn$convForward(%216, %arg138) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,768,16,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x384x16x16xf32>, tensor<768x384x1x1xf32>) -> tuple<tensor<8x768x16x16xf32>, tensor<0xui8>>
    %218 = stablehlo.get_tuple_element %217[0] : (tuple<tensor<8x768x16x16xf32>, tensor<0xui8>>) -> tensor<8x768x16x16xf32>
    %219 = "mhlo.fusion"(%arg140, %arg139, %218, %arg234, %arg235) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<8x768x16x16xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x768x16x16xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<768xf32>
      %303 = stablehlo.rsqrt %302 : tensor<768xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x768x16x16xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x768x16x16xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<8x768x16x16xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x768x16x16xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768x16x16xf32>
      %311 = stablehlo.negate %309 : tensor<8x768x16x16xf32>
      %312 = stablehlo.exponential %311 : tensor<8x768x16x16xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x768x16x16xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x768x16x16xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x768x16x16xf32>
      mhlo.return %315 : tensor<8x768x16x16xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x768x16x16xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<8x768x16x16xf32>
    %220 = stablehlo.custom_call @__cudnn$convForward(%219, %arg141) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,768,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<8x768x16x16xf32>, tensor<768x1x3x3xf32>) -> tuple<tensor<8x768x8x8xf32>, tensor<0xui8>>
    %221 = stablehlo.get_tuple_element %220[0] : (tuple<tensor<8x768x8x8xf32>, tensor<0xui8>>) -> tensor<8x768x8x8xf32>
    %222 = "mhlo.fusion"(%arg143, %arg142, %221, %arg236, %arg237) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<768xf32>, %arg246: tensor<768xf32>, %arg247: tensor<8x768x8x8xf32>, %arg248: tensor<768xf32>, %arg249: tensor<768xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x768x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<768xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<768xf32>
      %303 = stablehlo.rsqrt %302 : tensor<768xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x768x8x8xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x768x8x8xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<768xf32>) -> tensor<8x768x8x8xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x768x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x768x8x8xf32>
      %311 = stablehlo.negate %309 : tensor<8x768x8x8xf32>
      %312 = stablehlo.exponential %311 : tensor<8x768x8x8xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x768x8x8xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x768x8x8xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x768x8x8xf32>
      mhlo.return %315 : tensor<8x768x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<768xf32>, tensor<768xf32>, tensor<8x768x8x8xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<8x768x8x8xf32>
    %223 = stablehlo.custom_call @__cudnn$convForward(%222, %arg144) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<8x768x8x8xf32>, tensor<512x768x1x1xf32>) -> tuple<tensor<8x512x8x8xf32>, tensor<0xui8>>
    %224 = stablehlo.get_tuple_element %223[0] : (tuple<tensor<8x512x8x8xf32>, tensor<0xui8>>) -> tensor<8x512x8x8xf32>
    %225 = "mhlo.fusion"(%arg146, %arg145, %224, %arg238, %arg239) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<8x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<512xf32>
      %303 = stablehlo.rsqrt %302 : tensor<512xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x512x8x8xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x512x8x8xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x512x8x8xf32>
      mhlo.return %309 : tensor<8x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %226 = stablehlo.custom_call @__cudnn$convForward(%225, %arg147) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x8x8xf32>, tensor<512x1x3x3xf32>) -> tuple<tensor<8x512x8x8xf32>, tensor<0xui8>>
    %227 = stablehlo.get_tuple_element %226[0] : (tuple<tensor<8x512x8x8xf32>, tensor<0xui8>>) -> tensor<8x512x8x8xf32>
    %228 = "mhlo.fusion"(%arg149, %arg148, %227, %arg240, %arg241) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<8x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<512xf32>
      %303 = stablehlo.rsqrt %302 : tensor<512xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x512x8x8xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x512x8x8xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x512x8x8xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %310 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x512x8x8xf32>
      %311 = stablehlo.negate %309 : tensor<8x512x8x8xf32>
      %312 = stablehlo.exponential %311 : tensor<8x512x8x8xf32>
      %313 = stablehlo.add %310, %312 : tensor<8x512x8x8xf32>
      %314 = stablehlo.divide %310, %313 : tensor<8x512x8x8xf32>
      %315 = stablehlo.multiply %309, %314 : tensor<8x512x8x8xf32>
      mhlo.return %315 : tensor<8x512x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512x8x8xf32>
    %229 = stablehlo.custom_call @__cudnn$convForward(%228, %arg150) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x8x8xf32>, tensor<256x512x1x1xf32>) -> tuple<tensor<8x256x8x8xf32>, tensor<0xui8>>
    %230 = stablehlo.get_tuple_element %229[0] : (tuple<tensor<8x256x8x8xf32>, tensor<0xui8>>) -> tensor<8x256x8x8xf32>
    %231 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x8x8xf32>):
      %299 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %300 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %301 = stablehlo.multiply %299, %300 : tensor<4xi32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %303 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %304 = stablehlo.add %302, %303 : tensor<2x4xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %307 = stablehlo.compare LT, %305, %306 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %309 = stablehlo.add %305, %308 : tensor<1x1x2x4xi32>
      %310 = stablehlo.select %307, %309, %305 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %313 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %314 = stablehlo.compare LT, %304, %313 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %315 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %316 = stablehlo.add %304, %315 : tensor<2x4xi32>
      %317 = stablehlo.select %314, %316, %304 : tensor<2x4xi1>, tensor<2x4xi32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %319 = stablehlo.concatenate %312, %318, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %321 = "stablehlo.gather"(%arg245, %320) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x8x256x1x1xf32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x8x256x1x1xf32>) -> tensor<2x4x2x4x8x256xf32>
      %323 = stablehlo.transpose %322, dims = [4, 5, 0, 2, 1, 3] : (tensor<2x4x2x4x8x256xf32>) -> tensor<8x256x2x2x4x4xf32>
      mhlo.return %323 : tensor<8x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8x8xf32>) -> tensor<8x256x2x2x4x4xf32>
    %232 = "mhlo.fusion"(%230) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x8x8xf32>):
      %299 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %300 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %301 = stablehlo.multiply %299, %300 : tensor<4xi32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %303 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %304 = stablehlo.add %302, %303 : tensor<2x4xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %307 = stablehlo.compare LT, %305, %306 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %309 = stablehlo.add %305, %308 : tensor<1x1x2x4xi32>
      %310 = stablehlo.select %307, %309, %305 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %313 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %314 = stablehlo.compare LT, %304, %313 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %315 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %316 = stablehlo.add %304, %315 : tensor<2x4xi32>
      %317 = stablehlo.select %314, %316, %304 : tensor<2x4xi1>, tensor<2x4xi32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %319 = stablehlo.concatenate %312, %318, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %321 = "stablehlo.gather"(%arg245, %320) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2, 3, 4], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 256, 1, 1>}> : (tensor<8x256x8x8xf32>, tensor<64x2xi32>) -> tensor<64x8x256x1x1xf32>
      %322 = mhlo.bitcast %321 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<64x8x256x1x1xf32>) -> tensor<64x8x256xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %323 = stablehlo.reduce(%322 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x8x256xf32>, tensor<f32>) -> tensor<64x8xf32>
      mhlo.return %323 : tensor<64x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8x8xf32>) -> tensor<64x8xf32>
    %233 = "mhlo.fusion"(%232) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<64x8xf32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.reduce(%arg245 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<64x8xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %299 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<64x8xf32>) -> tensor<8xf32>
    %234:2 = "mhlo.fusion"(%231, %233) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x2x2x4x4xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x2x2x4x4xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x2x2x4x4xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x2x2x4x4xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x256x2x2x4x4xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x2x2x4x4xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x2x2x4x4xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x2x2x4x4xf32>)
    %235 = stablehlo.tuple %234#0, %234#1 {xla_shape = "(f32[8]{0}, f32[8,256,2,2,4,4]{5,4,3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x2x2x4x4xf32>>
    %236 = "mhlo.fusion"(%234#1, %arg151, %arg152, %234#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x2x2x4x4xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %300 = mhlo.bitcast %299 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<1x8x16384xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %302 = stablehlo.multiply %arg248, %301 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %303 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %304 = stablehlo.add %302, %303 : tensor<8xf32>
      %305 = stablehlo.rsqrt %304 : tensor<8xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0] : (tensor<8xf32>) -> tensor<8x256x2x2x4x4xf32>
      %307 = stablehlo.multiply %arg245, %306 : tensor<8x256x2x2x4x4xf32>
      %308 = mhlo.bitcast %307 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x256x2x2x4x4xf32>) -> tensor<1x8x16384xf32>
      %309 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<1x8x16384xf32>
      %311 = stablehlo.multiply %308, %310 : tensor<1x8x16384xf32>
      %312 = stablehlo.add %300, %311 : tensor<1x8x16384xf32>
      mhlo.return %312 : tensor<1x8x16384xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x2x2x4x4xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<1x8x16384xf32>
    %237 = mhlo.bitcast %236 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[2, 1, 0]> : tensor<3xindex>} : (tensor<1x8x16384xf32>) -> tensor<8x256x4x16xf32>
    %238 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%237, %arg153, %arg154) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<8x513x4x16xf32>, tensor<0xui8>>
    %239 = stablehlo.get_tuple_element %238[0] : (tuple<tensor<8x513x4x16xf32>, tensor<0xui8>>) -> tensor<8x513x4x16xf32>
    %240 = "mhlo.fusion"(%239) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x513x4x16xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 257:513, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x16xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x256x4x16xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:257, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x1x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x16xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x16xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x16xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x16xf32>) -> tensor<8x256x4x16xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x256x4x16xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x256x4x16xf32>, tensor<f32>) -> tensor<8x256x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x256x4xf32>) -> tensor<8x256x4x16xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x256x4x16xf32>
      mhlo.return %318 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %241 = mhlo.bitcast %231 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<8x256x2x2x4x4xf32>) -> tensor<8x256x4x16xf32>
    %242 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%240, %arg155, %arg156, %241) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<8x256x4x16xf32>) -> tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>
    %243 = stablehlo.get_tuple_element %242[0] : (tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>) -> tensor<8x256x4x16xf32>
    %244 = "mhlo.fusion"(%243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %300 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>) -> tensor<8xf32>
    %245:2 = "mhlo.fusion"(%243, %244) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x4x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>)
    %246 = stablehlo.tuple %245#0, %245#1 {xla_shape = "(f32[8]{0}, f32[8,256,4,16]{3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x4x16xf32>>
    %247 = "mhlo.fusion"(%245#1, %arg157, %arg158, %245#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %303 = stablehlo.add %301, %302 : tensor<8xf32>
      %304 = stablehlo.rsqrt %303 : tensor<8xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %306 = stablehlo.multiply %arg245, %305 : tensor<8x256x4x16xf32>
      %307 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<8x256x4x16xf32>
      %309 = stablehlo.add %299, %308 : tensor<8x256x4x16xf32>
      mhlo.return %309 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %248 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%247, %arg159, %arg160) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<8x512x4x16xf32>, tensor<0xui8>>
    %249 = stablehlo.get_tuple_element %248[0] : (tuple<tensor<8x512x4x16xf32>, tensor<0xui8>>) -> tensor<8x512x4x16xf32>
    %250 = "mhlo.fusion"(%249) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x4x16xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x512x4x16xf32>
      %301 = stablehlo.exponential %300 : tensor<8x512x4x16xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x512x4x16xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x512x4x16xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x512x4x16xf32>
      mhlo.return %304 : tensor<8x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x4x16xf32>) -> tensor<8x512x4x16xf32>
    %251 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%250, %arg161, %arg162, %243) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<8x256x4x16xf32>) -> tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>
    %252 = stablehlo.get_tuple_element %251[0] : (tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>) -> tensor<8x256x4x16xf32>
    %253 = "mhlo.fusion"(%252) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %300 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>) -> tensor<8xf32>
    %254:2 = "mhlo.fusion"(%252, %253) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x4x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>)
    %255 = stablehlo.tuple %254#0, %254#1 {xla_shape = "(f32[8]{0}, f32[8,256,4,16]{3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x4x16xf32>>
    %256 = "mhlo.fusion"(%254#1, %arg163, %arg164, %254#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %303 = stablehlo.add %301, %302 : tensor<8xf32>
      %304 = stablehlo.rsqrt %303 : tensor<8xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %306 = stablehlo.multiply %arg245, %305 : tensor<8x256x4x16xf32>
      %307 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<8x256x4x16xf32>
      %309 = stablehlo.add %299, %308 : tensor<8x256x4x16xf32>
      mhlo.return %309 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %257 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%256, %arg165, %arg166) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<8x513x4x16xf32>, tensor<0xui8>>
    %258 = stablehlo.get_tuple_element %257[0] : (tuple<tensor<8x513x4x16xf32>, tensor<0xui8>>) -> tensor<8x513x4x16xf32>
    %259 = "mhlo.fusion"(%258) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x513x4x16xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 257:513, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x16xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x256x4x16xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:257, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x1x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x16xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x16xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x16xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x16xf32>) -> tensor<8x256x4x16xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x256x4x16xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x256x4x16xf32>, tensor<f32>) -> tensor<8x256x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x256x4xf32>) -> tensor<8x256x4x16xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x256x4x16xf32>
      mhlo.return %318 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %260 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%259, %arg167, %arg168, %252) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<8x256x4x16xf32>) -> tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>
    %261 = stablehlo.get_tuple_element %260[0] : (tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>) -> tensor<8x256x4x16xf32>
    %262 = "mhlo.fusion"(%261) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %300 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>) -> tensor<8xf32>
    %263:2 = "mhlo.fusion"(%261, %262) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x4x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>)
    %264 = stablehlo.tuple %263#0, %263#1 {xla_shape = "(f32[8]{0}, f32[8,256,4,16]{3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x4x16xf32>>
    %265 = "mhlo.fusion"(%263#1, %arg169, %arg170, %263#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %303 = stablehlo.add %301, %302 : tensor<8xf32>
      %304 = stablehlo.rsqrt %303 : tensor<8xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %306 = stablehlo.multiply %arg245, %305 : tensor<8x256x4x16xf32>
      %307 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<8x256x4x16xf32>
      %309 = stablehlo.add %299, %308 : tensor<8x256x4x16xf32>
      mhlo.return %309 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %266 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%265, %arg171, %arg172) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<8x512x4x16xf32>, tensor<0xui8>>
    %267 = stablehlo.get_tuple_element %266[0] : (tuple<tensor<8x512x4x16xf32>, tensor<0xui8>>) -> tensor<8x512x4x16xf32>
    %268 = "mhlo.fusion"(%267) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x4x16xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x512x4x16xf32>
      %301 = stablehlo.exponential %300 : tensor<8x512x4x16xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x512x4x16xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x512x4x16xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x512x4x16xf32>
      mhlo.return %304 : tensor<8x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x4x16xf32>) -> tensor<8x512x4x16xf32>
    %269 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%268, %arg173, %arg174, %261) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<8x256x4x16xf32>) -> tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>
    %270 = stablehlo.get_tuple_element %269[0] : (tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>) -> tensor<8x256x4x16xf32>
    %271 = "mhlo.fusion"(%270) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %300 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>) -> tensor<8xf32>
    %272:2 = "mhlo.fusion"(%270, %271) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x4x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>)
    %273 = stablehlo.tuple %272#0, %272#1 {xla_shape = "(f32[8]{0}, f32[8,256,4,16]{3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x4x16xf32>>
    %274 = "mhlo.fusion"(%272#1, %arg175, %arg176, %272#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %303 = stablehlo.add %301, %302 : tensor<8xf32>
      %304 = stablehlo.rsqrt %303 : tensor<8xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %306 = stablehlo.multiply %arg245, %305 : tensor<8x256x4x16xf32>
      %307 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<8x256x4x16xf32>
      %309 = stablehlo.add %299, %308 : tensor<8x256x4x16xf32>
      mhlo.return %309 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %275 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%274, %arg177, %arg178) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,513,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<513x256x1x1xf32>, tensor<513xf32>) -> tuple<tensor<8x513x4x16xf32>, tensor<0xui8>>
    %276 = stablehlo.get_tuple_element %275[0] : (tuple<tensor<8x513x4x16xf32>, tensor<0xui8>>) -> tensor<8x513x4x16xf32>
    %277 = "mhlo.fusion"(%276) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x513x4x16xf32>):
      %299 = stablehlo.slice %arg245 [0:8, 257:513, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x4x16xf32>
      %301 = stablehlo.maximum %299, %300 : tensor<8x256x4x16xf32>
      %302 = stablehlo.slice %arg245 [0:8, 1:257, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
      %303 = stablehlo.slice %arg245 [0:8, 0:1, 0:4, 0:16] : (tensor<8x513x4x16xf32>) -> tensor<8x1x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
      %306 = stablehlo.broadcast_in_dim %305, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
      %307 = stablehlo.subtract %303, %306 : tensor<8x1x4x16xf32>
      %308 = stablehlo.exponential %307 : tensor<8x1x4x16xf32>
      %309 = mhlo.bitcast %308 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %310 = stablehlo.reduce(%309 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x4x16xf32>, tensor<f32>) -> tensor<8x4xf32>
      %311 = stablehlo.broadcast_in_dim %310, dims = [0, 2] : (tensor<8x4xf32>) -> tensor<8x1x4x16xf32>
      %312 = stablehlo.divide %308, %311 : tensor<8x1x4x16xf32>
      %313 = mhlo.bitcast %312 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x1x4x16xf32>) -> tensor<8x4x16xf32>
      %314 = stablehlo.broadcast_in_dim %313, dims = [0, 2, 3] : (tensor<8x4x16xf32>) -> tensor<8x256x4x16xf32>
      %315 = stablehlo.multiply %302, %314 : tensor<8x256x4x16xf32>
      %316 = stablehlo.reduce(%315 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x256x4x16xf32>, tensor<f32>) -> tensor<8x256x4xf32>
      %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<8x256x4xf32>) -> tensor<8x256x4x16xf32>
      %318 = stablehlo.multiply %301, %317 : tensor<8x256x4x16xf32>
      mhlo.return %318 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x513x4x16xf32>) -> tensor<8x256x4x16xf32>
    %278 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%277, %arg179, %arg180, %270) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<256x256x1x1xf32>, tensor<256xf32>, tensor<8x256x4x16xf32>) -> tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>
    %279 = stablehlo.get_tuple_element %278[0] : (tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>) -> tensor<8x256x4x16xf32>
    %280 = "mhlo.fusion"(%279) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %300 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>) -> tensor<8xf32>
    %281:2 = "mhlo.fusion"(%279, %280) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x4x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>)
    %282 = stablehlo.tuple %281#0, %281#1 {xla_shape = "(f32[8]{0}, f32[8,256,4,16]{3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x4x16xf32>>
    %283 = "mhlo.fusion"(%281#1, %arg181, %arg182, %281#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<256xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %303 = stablehlo.add %301, %302 : tensor<8xf32>
      %304 = stablehlo.rsqrt %303 : tensor<8xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %306 = stablehlo.multiply %arg245, %305 : tensor<8x256x4x16xf32>
      %307 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<8x256x4x16xf32>
      %309 = stablehlo.add %299, %308 : tensor<8x256x4x16xf32>
      mhlo.return %309 : tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<256xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<8x256x4x16xf32>
    %284 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%283, %arg183, %arg184) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x4x16xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tuple<tensor<8x512x4x16xf32>, tensor<0xui8>>
    %285 = stablehlo.get_tuple_element %284[0] : (tuple<tensor<8x512x4x16xf32>, tensor<0xui8>>) -> tensor<8x512x4x16xf32>
    %286 = "mhlo.fusion"(%285) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x512x4x16xf32>):
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512x4x16xf32>
      %300 = stablehlo.negate %arg245 : tensor<8x512x4x16xf32>
      %301 = stablehlo.exponential %300 : tensor<8x512x4x16xf32>
      %302 = stablehlo.add %299, %301 : tensor<8x512x4x16xf32>
      %303 = stablehlo.divide %299, %302 : tensor<8x512x4x16xf32>
      %304 = stablehlo.multiply %arg245, %303 : tensor<8x512x4x16xf32>
      mhlo.return %304 : tensor<8x512x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512x4x16xf32>) -> tensor<8x512x4x16xf32>
    %287 = stablehlo.custom_call @__cudnn$convBiasActivationForward(%286, %arg185, %arg186, %279) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:1,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,256,4,16]{3,2,1,0}, u8[0]{0})"} : (tensor<8x512x4x16xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<8x256x4x16xf32>) -> tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>
    %288 = stablehlo.get_tuple_element %287[0] : (tuple<tensor<8x256x4x16xf32>, tensor<0xui8>>) -> tensor<8x256x4x16xf32>
    %289 = "mhlo.fusion"(%288) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>):
      %299 = mhlo.bitcast %arg245 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %300 = stablehlo.reduce(%299 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %300 : tensor<8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>) -> tensor<8xf32>
    %290:2 = "mhlo.fusion"(%288, %289) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x4x16xf32>, %arg246: tensor<8xf32>):
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %300 = stablehlo.multiply %arg246, %299 : tensor<8xf32>
      %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %302 = stablehlo.subtract %arg245, %301 : tensor<8x256x4x16xf32>
      %303 = stablehlo.multiply %302, %302 : tensor<8x256x4x16xf32>
      %304 = mhlo.bitcast %303 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x16384xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %305 = stablehlo.reduce(%304 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16384xf32>, tensor<f32>) -> tensor<8xf32>
      mhlo.return %305, %302 : tensor<8xf32>, tensor<8x256x4x16xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x4x16xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8x256x4x16xf32>)
    %291 = stablehlo.tuple %290#0, %290#1 {xla_shape = "(f32[8]{0}, f32[8,256,4,16]{3,2,1,0})"} : tuple<tensor<8xf32>, tensor<8x256x4x16xf32>>
    %292 = "mhlo.fusion"(%arg187, %290#1, %arg188, %290#0) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<256xf32>, %arg246: tensor<8x256x4x16xf32>, %arg247: tensor<256xf32>, %arg248: tensor<8xf32>):
      %299 = stablehlo.broadcast_in_dim %arg247, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %cst = stablehlo.constant dense<6.10351563E-5> : tensor<f32>
      %300 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %301 = stablehlo.multiply %arg248, %300 : tensor<8xf32>
      %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %302 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %303 = stablehlo.add %301, %302 : tensor<8xf32>
      %304 = stablehlo.rsqrt %303 : tensor<8xf32>
      %305 = stablehlo.broadcast_in_dim %304, dims = [0] : (tensor<8xf32>) -> tensor<8x256x4x16xf32>
      %306 = stablehlo.multiply %arg246, %305 : tensor<8x256x4x16xf32>
      %307 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<256xf32>) -> tensor<8x256x4x16xf32>
      %308 = stablehlo.multiply %306, %307 : tensor<8x256x4x16xf32>
      %309 = stablehlo.add %299, %308 : tensor<8x256x4x16xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x256x4x16xf32>) -> tensor<8x256x2x2x4x4xf32>
      %311 = stablehlo.transpose %310, dims = [2, 4, 3, 5, 0, 1] : (tensor<8x256x2x2x4x4xf32>) -> tensor<2x4x2x4x8x256xf32>
      mhlo.return %311 : tensor<2x4x2x4x8x256xf32>
    }) {output_operand_aliasing = []} : (tensor<256xf32>, tensor<8x256x4x16xf32>, tensor<256xf32>, tensor<8xf32>) -> tensor<2x4x2x4x8x256xf32>
    %293 = "mhlo.fusion"(%0, %292) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<8x256x8x8xf32>, %arg246: tensor<2x4x2x4x8x256xf32>):
      %299 = stablehlo.iota dim = 0 : tensor<4xi32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %300 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
      %301 = stablehlo.multiply %299, %300 : tensor<4xi32>
      %302 = stablehlo.broadcast_in_dim %301, dims = [1] : (tensor<4xi32>) -> tensor<2x4xi32>
      %303 = stablehlo.iota dim = 0 : tensor<2x4xi32>
      %304 = stablehlo.add %302, %303 : tensor<2x4xi32>
      %305 = mhlo.bitcast %304 {result_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<2x4xi32>) -> tensor<1x1x2x4xi32>
      %c_0 = stablehlo.constant dense<0> : tensor<i32>
      %306 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %307 = stablehlo.compare LT, %305, %306 : (tensor<1x1x2x4xi32>, tensor<1x1x2x4xi32>) -> tensor<1x1x2x4xi1>
      %c_1 = stablehlo.constant dense<8> : tensor<i32>
      %308 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1x1x2x4xi32>
      %309 = stablehlo.add %305, %308 : tensor<1x1x2x4xi32>
      %310 = stablehlo.select %307, %309, %305 : tensor<1x1x2x4xi1>, tensor<1x1x2x4xi32>
      %311 = mhlo.bitcast %310 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<1x1x2x4xi32>) -> tensor<2x4xi32>
      %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %313 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %314 = stablehlo.compare LT, %304, %313 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi1>
      %315 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2x4xi32>
      %316 = stablehlo.add %304, %315 : tensor<2x4xi32>
      %317 = stablehlo.select %314, %316, %304 : tensor<2x4xi1>, tensor<2x4xi32>
      %318 = stablehlo.broadcast_in_dim %317, dims = [2, 3] : (tensor<2x4xi32>) -> tensor<2x4x2x4x1xi32>
      %319 = stablehlo.concatenate %312, %318, dim = 4 : (tensor<2x4x2x4x1xi32>, tensor<2x4x2x4x1xi32>) -> tensor<2x4x2x4x2xi32>
      %320 = mhlo.bitcast %319 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>} : (tensor<2x4x2x4x2xi32>) -> tensor<64x2xi32>
      %321 = mhlo.bitcast %arg246 {result_layout = dense<[4, 3, 2, 1, 0]> : tensor<5xindex>, source_layout = dense<[5, 4, 3, 2, 1, 0]> : tensor<6xindex>} : (tensor<2x4x2x4x8x256xf32>) -> tensor<64x8x256x1x1xf32>
      %322 = "stablehlo.scatter"(%arg245, %320, %321) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3, 4], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg247: tensor<f32>, %arg248: tensor<f32>):
        %323 = stablehlo.add %arg247, %arg248 : tensor<f32>
        stablehlo.return %323 : tensor<f32>
      }) : (tensor<8x256x8x8xf32>, tensor<64x2xi32>, tensor<64x8x256x1x1xf32>) -> tensor<8x256x8x8xf32>
      mhlo.return %322 : tensor<8x256x8x8xf32>
    }) {output_operand_aliasing = []} : (tensor<8x256x8x8xf32>, tensor<2x4x2x4x8x256xf32>) -> tensor<8x256x8x8xf32>
    %294 = stablehlo.custom_call @__cudnn$convForward(%293, %arg189) {backend_config = "{\22operation_queue_id\22:\220\22,\22wait_on_operation_queues\22:[],\22cudnn_conv_backend_config\22:{\22activation_mode\22:\22kNone\22,\22conv_result_scale\22:1,\22side_input_scale\22:0,\22leakyrelu_alpha\22:0},\22force_earliest_schedule\22:false,\22reification_cost\22:[],\22device_type\22:\22DEVICE_TYPE_INVALID\22}", xla_shape = "(f32[8,512,8,8]{3,2,1,0}, u8[0]{0})"} : (tensor<8x256x8x8xf32>, tensor<512x256x1x1xf32>) -> tuple<tensor<8x512x8x8xf32>, tensor<0xui8>>
    %295 = stablehlo.get_tuple_element %294[0] : (tuple<tensor<8x512x8x8xf32>, tensor<0xui8>>) -> tensor<8x512x8x8xf32>
    %296 = "mhlo.fusion"(%arg191, %arg190, %295, %arg242, %arg243) <{fusion_kind = #mhlo<fusion_kind kInput>}> ({
    ^bb0(%arg245: tensor<512xf32>, %arg246: tensor<512xf32>, %arg247: tensor<8x512x8x8xf32>, %arg248: tensor<512xf32>, %arg249: tensor<512xf32>):
      %299 = stablehlo.broadcast_in_dim %arg248, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %300 = stablehlo.subtract %arg247, %299 : tensor<8x512x8x8xf32>
      %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
      %301 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
      %302 = stablehlo.add %arg249, %301 : tensor<512xf32>
      %303 = stablehlo.rsqrt %302 : tensor<512xf32>
      %304 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %305 = stablehlo.multiply %300, %304 : tensor<8x512x8x8xf32>
      %306 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %307 = stablehlo.multiply %305, %306 : tensor<8x512x8x8xf32>
      %308 = stablehlo.broadcast_in_dim %arg245, dims = [1] : (tensor<512xf32>) -> tensor<8x512x8x8xf32>
      %309 = stablehlo.add %307, %308 : tensor<8x512x8x8xf32>
      %310 = mhlo.bitcast %309 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[3, 2, 1, 0]> : tensor<4xindex>} : (tensor<8x512x8x8xf32>) -> tensor<8x512x64xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %311 = stablehlo.reduce(%310 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<8x512x64xf32>, tensor<f32>) -> tensor<8x512xf32>
      mhlo.return %311 : tensor<8x512xf32>
    }) {output_operand_aliasing = []} : (tensor<512xf32>, tensor<512xf32>, tensor<8x512x8x8xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<8x512xf32>
    %297 = "mhlo.fusion"(%296, %arg192) <{fusion_kind = #mhlo<fusion_kind kCustom>}> ({
    ^bb0(%arg245: tensor<8x512xf32>, %arg246: tensor<2x512xf32>):
      %cst = stablehlo.constant dense<1.562500e-02> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x512xf32>
      %300 = stablehlo.multiply %arg245, %299 : tensor<8x512xf32>
      %301 = stablehlo.dot_general %300, %arg246, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x512xf32>, tensor<2x512xf32>) -> tensor<8x2xf32>
      mhlo.return %301 : tensor<8x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x512xf32>, tensor<2x512xf32>) -> tensor<8x2xf32>
    %298 = "mhlo.fusion"(%297, %arg193) <{fusion_kind = #mhlo<fusion_kind kLoop>}> ({
    ^bb0(%arg245: tensor<8x2xf32>, %arg246: tensor<2xf32>):
      %299 = stablehlo.broadcast_in_dim %arg246, dims = [1] : (tensor<2xf32>) -> tensor<8x2xf32>
      %300 = stablehlo.add %299, %arg245 : tensor<8x2xf32>
      mhlo.return %300 : tensor<8x2xf32>
    }) {output_operand_aliasing = []} : (tensor<8x2xf32>, tensor<2xf32>) -> tensor<8x2xf32>
    return %298 : tensor<8x2xf32>
  }
}
